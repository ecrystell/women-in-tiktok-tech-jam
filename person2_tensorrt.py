"""Optional direct TensorRT engine for Person 2's standalone FFN.

This module deliberately has no import-time TensorRT, CUDA Python, or engine
build side effects.  It is an *experiment* for the Tesla T4 path and is not
wired into ``OptimizedTransformerBlock``.  A caller prepares an engine outside
timed inference, runs a strict preflight, and then supplies a preallocated
output buffer to :meth:`TensorRTFFNEngine.run_into`.

The static TensorRT graph is:

    FP16 residual [tokens, d_model]
      -> identity-affine LayerNorm (FP32 statistics)
      -> FP16 Linear
      -> TensorRT GELU_ERF
      -> FP16 Linear
      -> FP16 update [tokens, d_model]

It intentionally produces only the FFN update.  Residual addition, padding
masking, and invalid-token zeroing remain in the existing validated PyTorch /
CUDA path.  ``GELU_ERF`` is required; there is deliberately no tanh-GELU
fallback.

TensorRT engines own copies of their weights.  Reuse an engine only through
``TensorRTFFNEngineCache.get_or_build`` (or ``get_if_prepared``), because its
cache key includes every parameter's data pointer and PyTorch version counter.
Builds, tactic selection, stream-context creation, and numerical preflight are
all setup work and must stay outside latency measurements.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import importlib
import math
import threading
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


__all__ = [
    "DEFAULT_WORKSPACE_BYTES",
    "TensorRTAvailability",
    "TensorRTBuildError",
    "TensorRTFFNCache",
    "TensorRTFFNEngine",
    "TensorRTFFNError",
    "TensorRTFFNKey",
    "TensorRTNumericalError",
    "TensorRTPreflightResult",
    "TensorRTRuntimeError",
    "TensorRTUnavailableError",
    "TensorRTUnsupportedError",
    "clear_engine_cache",
    "get_if_prepared",
    "prepare_fixed_shape_ffn",
    "probe_tensorrt",
    "try_prepare_fixed_shape_ffn",
]


# A fixed workspace limit makes tactic selection reproducible for a given
# TensorRT version and T4.  It is part of the engine-cache key.
DEFAULT_WORKSPACE_BYTES = 256 * 1024 * 1024
_EXPECTED_T4_CAPABILITY = (7, 5)


class TensorRTFFNError(RuntimeError):
    """Base class for errors that should select the native Person 2 fallback."""


class TensorRTUnavailableError(TensorRTFFNError):
    """TensorRT or cuda-python is not installed in the active environment."""


class TensorRTUnsupportedError(TensorRTFFNError):
    """The input, device, mode, or installed TensorRT API is unsupported."""


class TensorRTBuildError(TensorRTFFNError):
    """TensorRT could not create a valid fixed-shape engine."""


class TensorRTRuntimeError(TensorRTFFNError):
    """TensorRT could not enqueue the prepared engine on the PyTorch stream."""


class TensorRTNumericalError(TensorRTFFNError):
    """The optional engine failed the repository's strict numerical preflight."""


@dataclass(frozen=True)
class TensorRTAvailability:
    """Lazy-import status for direct TensorRT and cuda-python."""

    available: bool
    reason: Optional[str]
    tensorrt_version: Optional[str]
    cuda_python_version: Optional[str]


@dataclass(frozen=True)
class _TensorRTModules:
    trt: Any
    cudart: Any
    tensorrt_version: str
    cuda_python_version: str


@dataclass(frozen=True)
class TensorSignature:
    """Identity/version data for a TensorRT constant copied from a parameter."""

    name: str
    data_ptr: int
    version: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: str
    device_index: int


@dataclass(frozen=True)
class TensorRTFFNKey:
    """All properties that can make an engine or its copied constants stale."""

    device_index: int
    device_name: str
    capability: Tuple[int, int]
    tensorrt_version: str
    cuda_runtime_version: int
    tokens: int
    d_model: int
    ffn_dim: int
    dtype: str
    eps: float
    workspace_bytes: int
    t4_only: bool
    norm_weight: TensorSignature
    norm_bias: TensorSignature
    up_weight: TensorSignature
    up_bias: TensorSignature
    down_weight: TensorSignature
    down_bias: TensorSignature


@dataclass(frozen=True)
class TensorRTPreflightResult:
    """Strict OR-tolerance validation result for an untimed engine preflight."""

    passed: bool
    max_abs_error: float
    max_rel_error: float
    failed_elements: int
    total_elements: int


_MODULE_LOCK = threading.RLock()
_MODULES: Optional[_TensorRTModules] = None


def _load_modules() -> _TensorRTModules:
    """Import TensorRT and cuda-python only when the optional path is requested."""

    global _MODULES
    with _MODULE_LOCK:
        if _MODULES is not None:
            return _MODULES
        try:
            trt = importlib.import_module("tensorrt")
        except (ImportError, OSError) as error:
            raise TensorRTUnavailableError(
                "direct TensorRT Python bindings are unavailable; install a "
                "CUDA-12-compatible TensorRT package in the Colab runtime "
                "outside this repository"
            ) from error
        try:
            cuda_package = importlib.import_module("cuda")
            cudart = importlib.import_module("cuda.cudart")
        except (ImportError, OSError) as error:
            raise TensorRTUnavailableError(
                "cuda-python (the 'cuda.cudart' module) is unavailable; "
                "TensorRT pointer binding is intentionally not attempted"
            ) from error

        _MODULES = _TensorRTModules(
            trt=trt,
            cudart=cudart,
            tensorrt_version=str(getattr(trt, "__version__", "unknown")),
            cuda_python_version=str(
                getattr(cuda_package, "__version__", getattr(cudart, "__version__", "unknown"))
            ),
        )
        return _MODULES


def probe_tensorrt() -> TensorRTAvailability:
    """Report optional-backend availability without building an engine."""

    try:
        modules = _load_modules()
    except TensorRTUnavailableError as error:
        return TensorRTAvailability(False, str(error), None, None)
    return TensorRTAvailability(
        True,
        None,
        modules.tensorrt_version,
        modules.cuda_python_version,
    )


def _result_status(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) else result


def _status_ok(cudart: Any, status: Any) -> bool:
    success = getattr(getattr(cudart, "cudaError_t", object()), "cudaSuccess", 0)
    try:
        return int(status) == int(success)
    except (TypeError, ValueError):
        return status == success


def _check_cuda_result(cudart: Any, result: Any, operation: str) -> Tuple[Any, ...]:
    """Normalize cuda-python's tuple return convention and raise clearly."""

    values: Tuple[Any, ...]
    if isinstance(result, tuple):
        values = tuple(result[1:])
    else:
        values = ()
    status = _result_status(result)
    if not _status_ok(cudart, status):
        raise TensorRTRuntimeError(
            f"cuda-python {operation} failed with status {status!s}"
        )
    return values


def _cuda_runtime_version(cudart: Any) -> int:
    get_version = getattr(cudart, "cudaRuntimeGetVersion", None)
    if get_version is None:
        raise TensorRTUnsupportedError(
            "cuda-python does not expose cudaRuntimeGetVersion"
        )
    values = _check_cuda_result(cudart, get_version(), "cudaRuntimeGetVersion")
    if len(values) != 1:
        raise TensorRTUnsupportedError(
            "cuda-python returned an unexpected cudaRuntimeGetVersion result"
        )
    return int(values[0])


def _cuda_current_device(cudart: Any) -> int:
    get_device = getattr(cudart, "cudaGetDevice", None)
    if get_device is None:
        raise TensorRTUnsupportedError("cuda-python does not expose cudaGetDevice")
    values = _check_cuda_result(cudart, get_device(), "cudaGetDevice")
    if len(values) != 1:
        raise TensorRTUnsupportedError(
            "cuda-python returned an unexpected cudaGetDevice result"
        )
    return int(values[0])


def _peek_cuda_error(cudart: Any) -> None:
    peek = getattr(cudart, "cudaPeekAtLastError", None)
    if peek is not None:
        _check_cuda_result(cudart, peek(), "cudaPeekAtLastError")


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    return bool(is_compiling()) if is_compiling is not None else False


def _tensor_signature(name: str, tensor: torch.Tensor) -> TensorSignature:
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return TensorSignature(
        name=name,
        data_ptr=int(tensor.data_ptr()),
        version=int(getattr(tensor, "_version", -1)),
        shape=tuple(int(item) for item in tensor.shape),
        stride=tuple(int(item) for item in tensor.stride()),
        dtype=str(tensor.dtype),
        device_index=int(device_index),
    )


def _parameter_tensors(
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    return norm_weight, norm_bias, up_weight, up_bias, down_weight, down_bias


def _require_identity_affine(
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
) -> None:
    """Reject nonidentity norm parameters instead of silently changing semantics."""

    if not torch.equal(norm_weight, torch.ones_like(norm_weight)):
        raise TensorRTUnsupportedError(
            "TensorRT FFN experiment requires identity norm2.weight"
        )
    if torch.count_nonzero(norm_bias).item() != 0:
        raise TensorRTUnsupportedError(
            "TensorRT FFN experiment requires zero norm2.bias"
        )


def _validate_static_arguments(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    eps: float,
    *,
    t4_only: bool,
) -> Tuple[int, int, int, int, str, Tuple[int, int]]:
    """Validate the narrow experimental contract before TensorRT is imported."""

    if _is_compiling():
        raise TensorRTUnsupportedError(
            "TensorRT fixed-shape engines are eager-only until CUDA-graph "
            "output ownership is explicitly validated"
        )
    if torch.is_grad_enabled():
        raise TensorRTUnsupportedError(
            "TensorRT FFN is inference-only; disable gradients before prepare/run"
        )
    if not residual.is_cuda:
        raise TensorRTUnsupportedError("TensorRT FFN requires a CUDA residual tensor")
    if residual.dtype != torch.float16:
        raise TensorRTUnsupportedError("TensorRT FFN supports FP16 only")
    if residual.ndim != 2 or not residual.is_contiguous():
        raise TensorRTUnsupportedError(
            "TensorRT FFN requires a contiguous flattened [tokens, d_model] input"
        )
    if residual.shape[0] <= 0 or residual.shape[1] <= 0:
        raise TensorRTUnsupportedError("TensorRT FFN requires non-empty static shapes")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise TensorRTUnsupportedError("LayerNorm epsilon must be finite and positive")

    device_index = residual.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    device_index = int(device_index)
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device_index))
    if t4_only and capability != _EXPECTED_T4_CAPABILITY:
        raise TensorRTUnsupportedError(
            "TensorRT FFN experiment is restricted to T4/SM75; "
            f"received SM{capability[0]}{capability[1]}"
        )

    tokens, d_model = (int(residual.shape[0]), int(residual.shape[1]))
    parameters = _parameter_tensors(
        norm_weight, norm_bias, up_weight, up_bias, down_weight, down_bias
    )
    for parameter in parameters:
        if (
            not parameter.is_cuda
            or parameter.device.index != device_index
            or parameter.dtype != torch.float16
            or not parameter.is_contiguous()
        ):
            raise TensorRTUnsupportedError(
                "TensorRT FFN requires contiguous FP16 parameters on the input device"
            )
    if tuple(norm_weight.shape) != (d_model,) or tuple(norm_bias.shape) != (d_model,):
        raise TensorRTUnsupportedError("norm2 parameters do not match d_model")
    if up_weight.ndim != 2 or up_weight.shape[1] != d_model:
        raise TensorRTUnsupportedError("ffn_in.weight must have shape [ffn_dim, d_model]")
    ffn_dim = int(up_weight.shape[0])
    if tuple(up_bias.shape) != (ffn_dim,):
        raise TensorRTUnsupportedError("ffn_in.bias must have shape [ffn_dim]")
    if tuple(down_weight.shape) != (d_model, ffn_dim):
        raise TensorRTUnsupportedError(
            "ffn_out.weight must have shape [d_model, ffn_dim]"
        )
    if tuple(down_bias.shape) != (d_model,):
        raise TensorRTUnsupportedError("ffn_out.bias must have shape [d_model]")
    _require_identity_affine(norm_weight, norm_bias)
    return (
        device_index,
        tokens,
        d_model,
        ffn_dim,
        torch.cuda.get_device_name(device_index),
        capability,
    )


def _make_key(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    eps: float,
    *,
    workspace_bytes: int,
    t4_only: bool,
) -> Tuple[TensorRTFFNKey, _TensorRTModules]:
    if workspace_bytes <= 0:
        raise TensorRTUnsupportedError("TensorRT workspace_bytes must be positive")
    (
        device_index,
        tokens,
        d_model,
        ffn_dim,
        device_name,
        capability,
    ) = _validate_static_arguments(
        residual,
        norm_weight,
        norm_bias,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        eps,
        t4_only=t4_only,
    )
    modules = _load_modules()
    runtime_version = _cuda_runtime_version(modules.cudart)
    return (
        TensorRTFFNKey(
            device_index=device_index,
            device_name=device_name,
            capability=capability,
            tensorrt_version=modules.tensorrt_version,
            cuda_runtime_version=runtime_version,
            tokens=tokens,
            d_model=d_model,
            ffn_dim=ffn_dim,
            dtype=str(residual.dtype),
            eps=float(eps),
            workspace_bytes=int(workspace_bytes),
            t4_only=bool(t4_only),
            norm_weight=_tensor_signature("norm_weight", norm_weight),
            norm_bias=_tensor_signature("norm_bias", norm_bias),
            up_weight=_tensor_signature("up_weight", up_weight),
            up_bias=_tensor_signature("up_bias", up_bias),
            down_weight=_tensor_signature("down_weight", down_weight),
            down_bias=_tensor_signature("down_bias", down_bias),
        ),
        modules,
    )


def _require_modern_tensorrt_api(trt: Any) -> None:
    missing = []
    if not hasattr(trt, "Builder"):
        missing.append("Builder")
    activation_gelu = hasattr(
        getattr(trt, "ActivationType", object()), "GELU_ERF"
    )
    unary_gelu = hasattr(getattr(trt, "UnaryOperation", object()), "GELU_ERF")
    if not (activation_gelu or unary_gelu):
        missing.append("ActivationType.GELU_ERF")
    network_flags = getattr(trt, "NetworkDefinitionCreationFlag", object())
    strongly_typed = hasattr(network_flags, "STRONGLY_TYPED")
    explicit_batch = hasattr(network_flags, "EXPLICIT_BATCH")
    if not (strongly_typed or explicit_batch):
        missing.append("a strongly-typed or explicit-batch network flag")
    if not strongly_typed and not hasattr(
        getattr(trt, "BuilderFlag", object()), "FP16"
    ):
        missing.append("BuilderFlag.FP16")
    if missing:
        raise TensorRTUnsupportedError(
            "installed TensorRT lacks required direct APIs: " + ", ".join(missing)
        )


def _host_half(tensor: torch.Tensor) -> Any:
    """Copy a setup-time FP16 parameter into TensorRT's host constant format."""

    try:
        import numpy as np
    except ImportError as error:
        raise TensorRTUnavailableError(
            "NumPy is required by the direct TensorRT Python bindings"
        ) from error
    return np.ascontiguousarray(tensor.detach().contiguous().cpu().numpy())


def _require_layer(layer: Any, label: str) -> Any:
    if layer is None:
        raise TensorRTBuildError(f"TensorRT failed to create {label}")
    return layer


def _set_layer_precision(layer: Any, dtype: Any, label: str) -> None:
    """Require precision constraints instead of accepting an implicit downgrade."""

    try:
        layer.precision = dtype
        layer.set_output_type(0, dtype)
    except (AttributeError, TypeError, RuntimeError) as error:
        raise TensorRTUnsupportedError(
            f"installed TensorRT cannot constrain {label} precision"
        ) from error


def _add_constant(network: Any, tensor: torch.Tensor, label: str) -> Any:
    layer = _require_layer(
        network.add_constant(tuple(int(value) for value in tensor.shape), _host_half(tensor)),
        f"{label} constant",
    )
    return layer.get_output(0)


def _build_serialized_engine(
    key: TensorRTFFNKey,
    modules: _TensorRTModules,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
) -> Tuple[Any, Any, Any]:
    """Build one static TensorRT network; this is setup work only."""

    trt = modules.trt
    _require_modern_tensorrt_api(trt)
    logger = trt.Logger(trt.Logger.ERROR)
    try:
        builder = trt.Builder(logger)
        network_flags = trt.NetworkDefinitionCreationFlag
        strongly_typed = hasattr(network_flags, "STRONGLY_TYPED")
        if strongly_typed:
            flags = 1 << int(network_flags.STRONGLY_TYPED)
        else:
            flags = 1 << int(network_flags.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        if network is None:
            raise TensorRTBuildError("TensorRT failed to create a static FFN network")
        config = builder.create_builder_config()
        if config is None:
            raise TensorRTBuildError("TensorRT failed to create a builder config")
        if not strongly_typed:
            config.set_flag(trt.BuilderFlag.FP16)
            obey = getattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS", None)
            if obey is not None:
                config.set_flag(obey)
        if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, int(key.workspace_bytes)
            )
        elif hasattr(config, "max_workspace_size"):
            config.max_workspace_size = int(key.workspace_bytes)
        else:
            raise TensorRTUnsupportedError(
                "installed TensorRT cannot set a fixed workspace limit"
            )

        residual = network.add_input(
            "residual", trt.DataType.HALF, (key.tokens, key.d_model)
        )
        if residual is None:
            raise TensorRTBuildError("TensorRT failed to add the residual input")

        scale = _add_constant(network, norm_weight.reshape(1, key.d_model), "norm scale")
        bias = _add_constant(network, norm_bias.reshape(1, key.d_model), "norm bias")
        norm = _require_layer(
            network.add_normalization(residual, scale, bias, 1 << 1),
            "identity LayerNorm",
        )
        norm.epsilon = float(key.eps)
        if not hasattr(norm, "compute_precision"):
            raise TensorRTUnsupportedError(
                "installed TensorRT cannot request FP32 LayerNorm statistics"
            )
        norm.compute_precision = trt.DataType.FLOAT
        if not strongly_typed:
            _set_layer_precision(norm, trt.DataType.HALF, "LayerNorm output")

        up_matrix = _add_constant(network, up_weight, "up-projection weight")
        up = _require_layer(
            network.add_matrix_multiply(
                norm.get_output(0),
                trt.MatrixOperation.NONE,
                up_matrix,
                trt.MatrixOperation.TRANSPOSE,
            ),
            "up-projection GEMM",
        )
        if not strongly_typed:
            _set_layer_precision(up, trt.DataType.HALF, "up-projection GEMM")
        up_bias_tensor = _add_constant(network, up_bias.reshape(1, key.ffn_dim), "up bias")
        up_bias_add = _require_layer(
            network.add_elementwise(
                up.get_output(0), up_bias_tensor, trt.ElementWiseOperation.SUM
            ),
            "up-projection bias",
        )
        if not strongly_typed:
            _set_layer_precision(up_bias_add, trt.DataType.HALF, "up-projection bias")

        # GELU_ERF is TensorRT's erf-form GELU.  Do not replace it with the
        # faster GELU_TANH variant: exact-GELU is part of this repository's
        # correctness contract.
        if hasattr(getattr(trt, "ActivationType", object()), "GELU_ERF"):
            gelu = _require_layer(
                network.add_activation(
                    up_bias_add.get_output(0), trt.ActivationType.GELU_ERF
                ),
                "GELU_ERF",
            )
        else:
            gelu = _require_layer(
                network.add_unary(
                    up_bias_add.get_output(0), trt.UnaryOperation.GELU_ERF
                ),
                "GELU_ERF",
            )
        if not strongly_typed:
            _set_layer_precision(gelu, trt.DataType.HALF, "GELU_ERF")

        down_matrix = _add_constant(network, down_weight, "down-projection weight")
        down = _require_layer(
            network.add_matrix_multiply(
                gelu.get_output(0),
                trt.MatrixOperation.NONE,
                down_matrix,
                trt.MatrixOperation.TRANSPOSE,
            ),
            "down-projection GEMM",
        )
        if not strongly_typed:
            _set_layer_precision(down, trt.DataType.HALF, "down-projection GEMM")
        down_bias_tensor = _add_constant(
            network, down_bias.reshape(1, key.d_model), "down bias"
        )
        update = _require_layer(
            network.add_elementwise(
                down.get_output(0), down_bias_tensor, trt.ElementWiseOperation.SUM
            ),
            "down-projection bias",
        )
        if not strongly_typed:
            _set_layer_precision(update, trt.DataType.HALF, "down-projection bias")
        output = update.get_output(0)
        output.name = "update"
        network.mark_output(output)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise TensorRTBuildError("TensorRT returned no serialized fixed-shape engine")
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise TensorRTBuildError("TensorRT failed to deserialize its own engine")
        return logger, runtime, engine
    except TensorRTFFNError:
        raise
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise TensorRTBuildError(
            f"TensorRT fixed-shape FFN build failed: {type(error).__name__}: {error}"
        ) from error


class TensorRTFFNEngine:
    """One immutable-weight TensorRT FFN engine with setup-time stream contexts.

    The object is intentionally narrow.  ``run_into`` never allocates an
    output tensor, never builds tactics, and never synchronizes.  It requires
    the caller to have prepared the current PyTorch stream first.
    """

    def __init__(
        self,
        key: TensorRTFFNKey,
        modules: _TensorRTModules,
        logger: Any,
        runtime: Any,
        engine: Any,
    ) -> None:
        self.key = key
        self._modules = modules
        self._logger = logger
        self._runtime = runtime
        self._engine = engine
        self._contexts: Dict[int, Any] = {}
        self._context_lock = threading.RLock()
        self._preflight_result: Optional[TensorRTPreflightResult] = None

        if not hasattr(engine, "num_io_tensors"):
            raise TensorRTUnsupportedError(
                "TensorRT engine lacks v3 named-I/O support; TensorRT 10+ is required"
            )
        expected_names = {"residual", "update"}
        names = {
            str(engine.get_tensor_name(index))
            for index in range(int(engine.num_io_tensors))
        }
        if not expected_names.issubset(names):
            raise TensorRTBuildError(
                f"TensorRT engine I/O mismatch: expected {expected_names}, got {names}"
            )
        for name in expected_names:
            if engine.get_tensor_dtype(name) != modules.trt.DataType.HALF:
                raise TensorRTBuildError(
                    f"TensorRT engine {name} tensor is not FP16"
                )

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.key.device_index)

    @property
    def is_preflighted(self) -> bool:
        """Whether this immutable engine passed strict native equivalence."""

        return self._preflight_result is not None and self._preflight_result.passed

    def prepare_stream(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Create a v3 execution context for ``stream`` outside timed inference."""

        with torch.cuda.device(self.key.device_index):
            chosen = stream or torch.cuda.current_stream(self.key.device_index)
            stream_handle = int(chosen.cuda_stream)
            if stream_handle == 0:
                raise TensorRTUnsupportedError(
                    "TensorRT requires a concrete CUDA stream handle"
                )
            with self._context_lock:
                if stream_handle in self._contexts:
                    return
                current_device = _cuda_current_device(self._modules.cudart)
                if current_device != self.key.device_index:
                    raise TensorRTUnsupportedError(
                        "cuda-python current device does not match the TensorRT engine device"
                    )
                context = self._engine.create_execution_context()
                if context is None:
                    raise TensorRTBuildError(
                        "TensorRT failed to create an execution context"
                    )
                if not hasattr(context, "execute_async_v3") or not hasattr(
                    context, "set_tensor_address"
                ):
                    raise TensorRTUnsupportedError(
                        "TensorRT v3 named-tensor execution APIs are required"
                    )
                set_shape = getattr(context, "set_input_shape", None)
                if set_shape is not None:
                    accepted = set_shape(
                        "residual", (self.key.tokens, self.key.d_model)
                    )
                    if accepted is False:
                        raise TensorRTBuildError(
                            "TensorRT rejected the static residual input shape"
                        )
                self._contexts[stream_handle] = context

    def _validate_runtime_tensors(
        self, residual: torch.Tensor, update: torch.Tensor
    ) -> int:
        if _is_compiling():
            raise TensorRTUnsupportedError(
                "TensorRT FFN cannot execute under torch.compile"
            )
        if torch.is_grad_enabled():
            raise TensorRTUnsupportedError(
                "TensorRT FFN is inference-only; disable gradients before run"
            )
        expected_shape = (self.key.tokens, self.key.d_model)
        for name, tensor in (("residual", residual), ("update", update)):
            if (
                not tensor.is_cuda
                or tensor.device.index != self.key.device_index
                or tensor.dtype != torch.float16
                or tuple(tensor.shape) != expected_shape
                or not tensor.is_contiguous()
            ):
                raise TensorRTUnsupportedError(
                    f"{name} must be a contiguous FP16 {expected_shape} tensor on "
                    f"cuda:{self.key.device_index}"
                )
        if int(residual.data_ptr()) == int(update.data_ptr()):
            raise TensorRTUnsupportedError(
                "TensorRT update output must not alias its residual input"
            )
        stream = torch.cuda.current_stream(self.key.device_index)
        return int(stream.cuda_stream)

    def _run_into_unchecked(
        self, residual: torch.Tensor, update: torch.Tensor
    ) -> torch.Tensor:
        """Enqueue after setup/preflight has established this engine is usable."""

        stream_handle = self._validate_runtime_tensors(residual, update)
        with self._context_lock:
            context = self._contexts.get(stream_handle)
            if context is None:
                raise TensorRTUnsupportedError(
                    "current CUDA stream was not prepared; call prepare_stream "
                    "outside timed inference"
                )
            if context.set_tensor_address("residual", int(residual.data_ptr())) is False:
                raise TensorRTRuntimeError("TensorRT rejected the residual pointer")
            if context.set_tensor_address("update", int(update.data_ptr())) is False:
                raise TensorRTRuntimeError("TensorRT rejected the update pointer")
            enqueued = context.execute_async_v3(stream_handle)
            if enqueued is False:
                raise TensorRTRuntimeError("TensorRT execute_async_v3 returned False")
            _peek_cuda_error(self._modules.cudart)
        return update

    def run_into(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        """Enqueue the fixed-shape FFN and write its update into ``update``.

        This does not allocate or synchronize.  The caller must call
        :meth:`prepare_stream` on the current PyTorch stream during setup.
        TensorRT enqueues on that same stream, so subsequent PyTorch operations
        on the stream observe the output without an explicit synchronization.
        """

        if not self.is_preflighted:
            raise TensorRTUnsupportedError(
                "TensorRT engine has not passed strict_preflight; use the native "
                "fallback until setup validates it"
            )
        return self._run_into_unchecked(residual, update)

    def strict_preflight(
        self,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        up_weight: torch.Tensor,
        up_bias: torch.Tensor,
        down_weight: torch.Tensor,
        down_bias: torch.Tensor,
        *,
        atol: float = 0.001,
        rtol: float = 0.01,
    ) -> TensorRTPreflightResult:
        """Compare this engine with the native exact-GELU FFN outside timing.

        A failed result raises :class:`TensorRTNumericalError` so callers can
        retain the native fallback without ever timing an incorrect engine.
        """

        if not self.matches(
            residual,
            norm_weight,
            norm_bias,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
        ):
            raise TensorRTUnsupportedError(
                "TensorRT engine key is stale for the supplied shape, device, "
                "or parameter versions; rebuild outside timed inference"
            )
        if atol < 0.0 or rtol < 0.0:
            raise TensorRTUnsupportedError("strict preflight tolerances must be nonnegative")
        update = torch.empty_like(residual)
        self._run_into_unchecked(residual, update)
        reference = F.linear(
            F.gelu(
                F.linear(
                    F.layer_norm(
                        residual,
                        (self.key.d_model,),
                        norm_weight,
                        norm_bias,
                        self.key.eps,
                    ),
                    up_weight,
                    up_bias,
                ),
                approximate="none",
            ),
            down_weight,
            down_bias,
        )
        error = (update.float() - reference.float()).abs()
        denominator = reference.float().abs()
        relative = torch.where(
            denominator > 0,
            error / denominator,
            torch.zeros_like(error),
        )
        passed = torch.isfinite(update) & torch.isfinite(reference)
        passed &= (error <= atol) | (error <= rtol * denominator)
        failed = int((~passed).sum().item())
        result = TensorRTPreflightResult(
            passed=failed == 0,
            max_abs_error=float(error.max().item()),
            max_rel_error=float(relative.max().item()),
            failed_elements=failed,
            total_elements=int(error.numel()),
        )
        if not result.passed:
            raise TensorRTNumericalError(
                "TensorRT strict preflight failed: "
                f"max_abs={result.max_abs_error:.6g}, "
                f"max_rel={result.max_rel_error:.6g}, "
                f"failed={result.failed_elements}/{result.total_elements}"
            )
        self._preflight_result = result
        return result

    def matches(
        self,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        up_weight: torch.Tensor,
        up_bias: torch.Tensor,
        down_weight: torch.Tensor,
        down_bias: torch.Tensor,
    ) -> bool:
        """Return whether the engine still matches live parameters and shape.

        This check is intentionally for setup/preflight; it can inspect tensors
        and must not be inserted into a latency-critical inference loop.
        """

        try:
            candidate, _ = _make_key(
                residual,
                norm_weight,
                norm_bias,
                up_weight,
                up_bias,
                down_weight,
                down_bias,
                self.key.eps,
                workspace_bytes=self.key.workspace_bytes,
                t4_only=self.key.t4_only,
            )
        except TensorRTFFNError:
            return False
        return candidate == self.key


class TensorRTFFNCache:
    """Small LRU cache of immutable-weight fixed-shape TensorRT engines.

    Cache misses build an engine and must be handled during setup.  The cache
    does not serialize engines to disk and therefore does not create generated
    repository artifacts.
    """

    def __init__(self, max_entries: int = 8) -> None:
        if max_entries <= 0:
            raise ValueError("TensorRTFFNCache.max_entries must be positive")
        self._max_entries = int(max_entries)
        self._engines: "OrderedDict[TensorRTFFNKey, TensorRTFFNEngine]" = OrderedDict()
        self._lock = threading.RLock()
        self.last_error: Optional[str] = None

    def clear(self) -> None:
        """Release Python references to cached TensorRT engines and contexts."""

        with self._lock:
            self._engines.clear()
            self.last_error = None

    def get_or_build(
        self,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        up_weight: torch.Tensor,
        up_bias: torch.Tensor,
        down_weight: torch.Tensor,
        down_bias: torch.Tensor,
        eps: float,
        *,
        workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
        t4_only: bool = True,
    ) -> TensorRTFFNEngine:
        """Return a prepared static engine, building it only on a cache miss.

        Call this outside timing and immediately call ``strict_preflight``.
        ``run_into`` requires a preallocated update buffer and does not call
        this method itself.
        """

        try:
            key, modules = _make_key(
                residual,
                norm_weight,
                norm_bias,
                up_weight,
                up_bias,
                down_weight,
                down_bias,
                eps,
                workspace_bytes=workspace_bytes,
                t4_only=t4_only,
            )
            with self._lock:
                cached = self._engines.get(key)
                if cached is not None:
                    self._engines.move_to_end(key)
                    cached.prepare_stream()
                    self.last_error = None
                    return cached

            # TensorRT tactic selection is device-specific, so ensure the
            # residual's CUDA device is current before the builder runs.
            with torch.cuda.device(key.device_index):
                logger, runtime, engine = _build_serialized_engine(
                    key,
                    modules,
                    norm_weight,
                    norm_bias,
                    up_weight,
                    up_bias,
                    down_weight,
                    down_bias,
                )
                candidate = TensorRTFFNEngine(key, modules, logger, runtime, engine)
                candidate.prepare_stream()
            with self._lock:
                existing = self._engines.get(key)
                if existing is not None:
                    # Another setup thread built the identical immutable engine
                    # while this one was compiling tactics.  Prefer its cache
                    # entry and release this temporary candidate naturally.
                    self._engines.move_to_end(key)
                    existing.prepare_stream()
                    self.last_error = None
                    return existing
                self._engines[key] = candidate
                self._engines.move_to_end(key)
                while len(self._engines) > self._max_entries:
                    self._engines.popitem(last=False)
                self.last_error = None
                return candidate
        except TensorRTFFNError as error:
            with self._lock:
                self.last_error = f"{type(error).__name__}: {error}"
            raise

    def get_if_prepared(
        self,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        up_weight: torch.Tensor,
        up_bias: torch.Tensor,
        down_weight: torch.Tensor,
        down_bias: torch.Tensor,
        eps: float,
        *,
        workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
        t4_only: bool = True,
    ) -> Optional[TensorRTFFNEngine]:
        """Return an already-built matching engine without building or allocating."""

        try:
            key, _ = _make_key(
                residual,
                norm_weight,
                norm_bias,
                up_weight,
                up_bias,
                down_weight,
                down_bias,
                eps,
                workspace_bytes=workspace_bytes,
                t4_only=t4_only,
            )
        except TensorRTFFNError as error:
            with self._lock:
                self.last_error = f"{type(error).__name__}: {error}"
            return None
        with self._lock:
            engine = self._engines.get(key)
            if engine is not None and engine.is_preflighted:
                self._engines.move_to_end(key)
                return engine
            return None

    def cache_size(self) -> int:
        with self._lock:
            return len(self._engines)


_DEFAULT_CACHE = TensorRTFFNCache()


def prepare_fixed_shape_ffn(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    eps: float,
    *,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    t4_only: bool = True,
) -> TensorRTFFNEngine:
    """Build/cache a direct TensorRT FFN engine during untimed setup.

    The returned engine has already passed an untimed strict native preflight
    and emits the update before residual addition and masking.  Use
    ``engine.run_into(residual, preallocated_update)`` only in eager FP16
    inference on the prepared stream.
    """

    engine = _DEFAULT_CACHE.get_or_build(
        residual,
        norm_weight,
        norm_bias,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        eps,
        workspace_bytes=workspace_bytes,
        t4_only=t4_only,
    )
    if not engine.is_preflighted:
        engine.strict_preflight(
            residual,
            norm_weight,
            norm_bias,
            up_weight,
            up_bias,
            down_weight,
            down_bias,
        )
    return engine


def try_prepare_fixed_shape_ffn(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    eps: float,
    *,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    t4_only: bool = True,
) -> Tuple[Optional[TensorRTFFNEngine], Optional[str]]:
    """Setup-friendly fallback wrapper that never raises optional-backend errors."""

    try:
        return (
            prepare_fixed_shape_ffn(
                residual,
                norm_weight,
                norm_bias,
                up_weight,
                up_bias,
                down_weight,
                down_bias,
                eps,
                workspace_bytes=workspace_bytes,
                t4_only=t4_only,
            ),
            None,
        )
    except TensorRTFFNError as error:
        return None, f"{type(error).__name__}: {error}"


def get_if_prepared(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    eps: float,
    *,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    t4_only: bool = True,
) -> Optional[TensorRTFFNEngine]:
    """Return a matching default-cache engine without triggering a TensorRT build."""

    return _DEFAULT_CACHE.get_if_prepared(
        residual,
        norm_weight,
        norm_bias,
        up_weight,
        up_bias,
        down_weight,
        down_bias,
        eps,
        workspace_bytes=workspace_bytes,
        t4_only=t4_only,
    )


def clear_engine_cache() -> None:
    """Drop all in-memory default-cache engines; no serialized artifacts exist."""

    _DEFAULT_CACHE.clear()
