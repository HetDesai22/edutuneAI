"""
Hardware detection.

Decides whether we run on GPU (CUDA) or CPU, which precision to use and whether
4-bit quantisation (QLoRA) is possible. Never fails silently: when CUDA is not
available a clear warning is printed.
"""

from __future__ import annotations

import importlib.util
import platform
import warnings
from dataclasses import dataclass, asdict

import torch

OOM_HELP = (
    "CUDA ran out of memory. Reduce memory usage in config.py:\n"
    "  - BATCH_SIZE = 1                 (and raise GRADIENT_ACCUMULATION_STEPS to keep the effective batch)\n"
    "  - MAX_SEQ_LENGTH = 512           (shorter sequences use far less activation memory)\n"
    "  - GRADIENT_CHECKPOINTING = True\n"
    "  - USE_4BIT = True                (QLoRA: 4-bit frozen base model)\n"
    "  - LORA_R = 8                     (fewer trainable parameters)\n"
    "  - MAX_NEW_TOKENS lower           (for generation)\n"
    "  - or choose a smaller MODEL_NAME, e.g. 'Qwen/Qwen2.5-0.5B-Instruct'.\n"
    "Also close other programs that use the GPU."
)


@dataclass
class DeviceInfo:
    cuda_available: bool
    device: str
    gpu_name: str | None
    gpu_memory_gb: float | None
    bf16_supported: bool
    bitsandbytes_available: bool
    torch_version: str
    cuda_version: str | None
    platform: str

    def to_dict(self) -> dict:
        return asdict(self)


def bitsandbytes_available() -> bool:
    """bitsandbytes provides the 4-bit kernels needed for QLoRA."""
    return importlib.util.find_spec("bitsandbytes") is not None


def get_device_info() -> DeviceInfo:
    cuda = torch.cuda.is_available()
    name, mem = None, None
    bf16 = False
    if cuda:
        props = torch.cuda.get_device_properties(0)
        name = props.name
        mem = round(props.total_memory / 1024**3, 2)
        bf16 = torch.cuda.is_bf16_supported()
    return DeviceInfo(
        cuda_available=cuda,
        device="cuda" if cuda else "cpu",
        gpu_name=name,
        gpu_memory_gb=mem,
        bf16_supported=bf16,
        bitsandbytes_available=bitsandbytes_available(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        platform=platform.platform(),
    )


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_compute_dtype() -> torch.dtype:
    """bfloat16 on modern GPUs, float16 on older GPUs, float32 on CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def can_use_4bit() -> bool:
    """4-bit loading needs a CUDA GPU and the bitsandbytes library."""
    return torch.cuda.is_available() and bitsandbytes_available()


def print_device_info() -> DeviceInfo:
    info = get_device_info()
    print("=" * 60)
    print("HARDWARE")
    print("=" * 60)
    print(f"CUDA Available: {info.cuda_available}")
    if info.cuda_available:
        print(f"GPU: {info.gpu_name}")
        print(f"GPU Memory: {info.gpu_memory_gb} GB")
        print(f"CUDA (torch build): {info.cuda_version}")
        print(f"bfloat16 supported: {info.bf16_supported}")
        print(f"bitsandbytes (4-bit) available: {info.bitsandbytes_available}")
    else:
        warnings.warn(
            "CUDA is NOT available - running on CPU. Inference will be slow and "
            "QLoRA 4-bit training is not possible on CPU. Install a CUDA build of "
            "PyTorch (see README) to use your NVIDIA GPU.",
            stacklevel=2,
        )
        print("WARNING: CUDA is NOT available. Using CPU.")
    print(f"PyTorch: {info.torch_version}")
    print("=" * 60)
    return info


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


if __name__ == "__main__":
    print_device_info()
