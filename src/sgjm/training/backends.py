from __future__ import annotations

import importlib
import platform
from typing import Literal

Backend = Literal["auto", "cuda", "rocm", "mlx", "cpu"]
ResolvedBackend = Literal["cuda", "rocm", "mlx", "cpu"]

_VALID = {"cuda", "rocm", "mlx", "cpu"}


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _torch_is_rocm() -> bool:
    try:
        import torch
    except Exception:
        return False
    hip = getattr(torch.version, "hip", None)
    return bool(hip)


def _torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect_backend() -> ResolvedBackend:
    if platform.system() == "Darwin" and platform.machine() == "arm64" and _has_module("mlx.core"):
        return "mlx"
    if _torch_cuda_available():
        return "rocm" if _torch_is_rocm() else "cuda"
    return "cpu"


def resolve_backend(requested: str) -> ResolvedBackend:
    if requested == "auto":
        return detect_backend()
    if requested not in _VALID:
        raise ValueError(f"unknown backend {requested!r}; expected one of {_VALID | {'auto'}}")
    return requested  # type: ignore[return-value]


def torch_device(backend: ResolvedBackend) -> str:
    # PyTorch ROCm builds expose ROCm devices through the cuda namespace.
    if backend in ("cuda", "rocm"):
        return "cuda"
    if backend == "cpu":
        return "cpu"
    raise ValueError(f"backend {backend!r} is not a torch backend")


def is_torch_backend(backend: ResolvedBackend) -> bool:
    return backend in ("cuda", "rocm", "cpu")


def is_mlx_backend(backend: ResolvedBackend) -> bool:
    return backend == "mlx"
