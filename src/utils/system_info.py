"""Phase 1: portable system and storage inspection."""

from __future__ import annotations
import os
import platform
import shutil
from pathlib import Path

def _bytes_to_gb(value: int) -> float:
    return round(value / (1024 ** 3), 2)

def detect_torch_device() -> str:
    try:
        import torch
    except Exception:
        return "python-only"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_system_info(project_root: str | Path = ".") -> dict:
    root = Path(project_root).resolve()
    disk = shutil.disk_usage(root)
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "project_root": str(root),
        "disk_total_gb": _bytes_to_gb(disk.total),
        "disk_free_gb": _bytes_to_gb(disk.free),
        "torch_device": detect_torch_device(),
    }

if __name__ == "__main__":
    for key, value in get_system_info().items():
        print(f"{key}: {value}")