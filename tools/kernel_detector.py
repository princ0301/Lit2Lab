import sys
import subprocess
from pathlib import Path
 
VENV_DIR = Path(".agent_venv")
VENV_KERNEL_NAME = "agent_venv"


def _get_python(venv_dir: Path) -> str:
    """Returns the python executable path inside the venv."""
    if sys.platform == "win32":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def _uv_available() -> bool:
    """Check if uv is installed."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _create_venv_uv(venv_dir: Path) -> bool:
    """Create venv using uv."""
    print(f"[kernel_detector] Creating venv with uv at {venv_dir} ...")
    result = subprocess.run(
        ["uv", "venv", str(venv_dir)],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode == 0:
        print("[kernel_detector] venv created with uv")
        return True
    print(f"[kernel_detector] uv venv failed: {result.stderr.strip()}")
    return False


def _create_venv_stdlib(venv_dir: Path) -> bool:
    """Fallback: create venv using stdlib venv module (includes pip by default)."""
    print(f"[kernel_detector] Creating venv with stdlib at {venv_dir} ...")
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode == 0:
        print("[kernel_detector] venv created with stdlib venv")
        return True
    print(f"[kernel_detector] stdlib venv failed: {result.stderr.strip()}")
    return False


def _install_ipykernel(venv_dir: Path) -> bool:
    """
    Install ipykernel inside the venv.
    Uses 'uv pip install' first — works even when pip is not bundled in the venv.
    Falls back to 'python -m pip' for stdlib venvs.
    """
    print("[kernel_detector] Installing ipykernel into venv ...")
 
    if _uv_available():
        result = subprocess.run(
            ["uv", "pip", "install", "ipykernel", "--python", str(venv_dir)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print("[kernel_detector] ipykernel installed via uv pip")
            return True
        print(f"[kernel_detector] uv pip failed: {result.stderr.strip()} → trying pip fallback")
 
    python_path = _get_python(venv_dir)
    result = subprocess.run(
        [python_path, "-m", "pip", "install", "ipykernel", "-q"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        print("[kernel_detector] ipykernel installed via pip")
        return True

    print(f"[kernel_detector] ipykernel install failed: {result.stderr.strip()}")
    return False


def _register_kernel(venv_dir: Path, kernel_name: str) -> bool:
    """Register the venv as a Jupyter kernel."""
    python_path = _get_python(venv_dir)
    print(f"[kernel_detector] Registering kernel '{kernel_name}' ...")
    result = subprocess.run(
        [python_path, "-m", "ipykernel", "install", "--user",
         "--name", kernel_name,
         "--display-name", f"Python ({kernel_name})"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        print(f"[kernel_detector] Kernel '{kernel_name}' registered")
        return True
    print(f"[kernel_detector] Kernel registration failed: {result.stderr.strip()}")
    return False


def setup_agent_venv() -> str:
    """
    Creates a fresh virtual environment for this agent session (if not already present),
    installs ipykernel, registers it as a Jupyter kernel, and returns the kernel name.

    Priority:
        1. uv venv + uv pip install  (fast, no pip needed in venv)
        2. stdlib venv + python -m pip (fallback)
        3. system python3            (last resort)

    Returns:
        kernel_name string to pass to nbconvert / notebook metadata
    """
    venv_dir = VENV_DIR
    python_path = _get_python(venv_dir)
 
    if venv_dir.exists() and Path(python_path).exists():
        print(f"[kernel_detector] Reusing existing agent venv at {venv_dir}")
        return VENV_KERNEL_NAME
 
    created = False
    if _uv_available():
        created = _create_venv_uv(venv_dir)
    if not created:
        created = _create_venv_stdlib(venv_dir)

    if not created:
        print("[kernel_detector]  Could not create venv → falling back to system python3")
        return "python3"
 
    if not _install_ipykernel(venv_dir):
        print("[kernel_detector]  ipykernel install failed → falling back to system python3")
        return "python3"
 
    if not _register_kernel(venv_dir, VENV_KERNEL_NAME):
        print("[kernel_detector]  Kernel registration failed → falling back to system python3")
        return "python3"

    return VENV_KERNEL_NAME


def get_agent_python() -> str:
    """
    Returns the python executable inside the agent venv.
    Used by script_runner to run generated scripts in the correct environment.
    """
    python = _get_python(VENV_DIR)
    if Path(python).exists():
        return python
    return sys.executable
 
_KERNEL_NAME = setup_agent_venv()

def detect_best_kernel() -> str:
     
    return _KERNEL_NAME