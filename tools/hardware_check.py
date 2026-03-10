import platform
import subprocess

def get_hardware_info() -> dict:
    """
    Returns a dict with CPU, RAM, and GPU info.
    Works on Windows, macOS, Linux.
    """
    info = {
        "cpu": "Unknown",
        "ram_gb": "?",
        "has_gpu": False,
        "gpu_name": None,
    }

    info["cpu"] = platform.processor() or platform.machine() or "Unknow"

    try:
        import os
        if platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=5
            )
            lines = [l.strip() for l in result.stdout().splitlines() if l.strip().isdigit()]
            if lines:
                info["ram_gb"] = round(int(lines[0]) / (1024 ** 3), 1)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / (1024 ** 2), 1)
                        break

    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            info["has_gpu"] = True
            info["gpu_name"] = result.stdout.strip().splitlines()[0]
            return info
    except (FileNotFoundError, Exception):
        pass

    try:
        import torch
        if torch.cuda.is_available():
            info["has_gpu"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            return info
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["has_gpu"] = True
            info["gpu_name"] = "Apple MPS (Metal)"
            return info
    except ImportError:
        pass

    return info

if __name__ == "__main__":
    print(get_hardware_info())