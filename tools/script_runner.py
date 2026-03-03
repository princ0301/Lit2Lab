import os
import sys
import subprocess
from typing import Tuple
from tools.kernel_detector import get_agent_python, VENV_DIR, _uv_available


def _ensure_pip_available(python: str) -> None:
    """
    uv venv doesn't bundle pip. Bootstrap it via 'uv pip install pip'
    so scripts that use 'python -m pip' work correctly.
    """
    result = subprocess.run(
        [python, "-m", "pip", "--version"],
        capture_output=True, timeout=10
    )
    if result.returncode == 0:
        return  # pip already available

    print("[script_runner] pip not found in venv — bootstrapping via uv pip ...")
    if _uv_available():
        r = subprocess.run(
            ["uv", "pip", "install", "pip", "--python", str(VENV_DIR)],
            capture_output=True, text=True, timeout=60
        )
        if r.returncode == 0:
            print("[script_runner] ✅ pip bootstrapped successfully")
        else:
            print(f"[script_runner] ⚠️  pip bootstrap failed: {r.stderr.strip()}")
    else:
        print("[script_runner] ⚠️  uv not available, cannot bootstrap pip")


def run_script(script_path: str, timeout: int = 300) -> Tuple[bool, list, str]:
    """
    Executes a Python script inside the agent venv and captures all output.

    Returns:
        (is_valid, errors, full_output)
        - is_valid: True if script exited with code 0
        - errors: list of error strings
        - full_output: combined stdout + stderr for LLM context
    """
    python = get_agent_python()

    # Make sure pip is available inside the venv before running any script
    _ensure_pip_available(python)

    print(f"[script_runner] Using python: {python}")
    print(f"[script_runner] Running: {script_path}")

    try:
        result = subprocess.run(
            [python, script_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        full_output = ""
        if stdout:
            full_output += f"--- STDOUT ---\n{stdout}\n"
        if stderr:
            full_output += f"--- STDERR ---\n{stderr}\n"

        if result.returncode == 0:
            print(f"[script_runner] ✅ Script exited cleanly (code 0)")
            return True, [], full_output
        else:
            print(f"[script_runner] ❌ Script exited with code {result.returncode}")
            errors = [stderr] if stderr else [f"Exit code {result.returncode} with no stderr"]
            return False, errors, full_output

    except subprocess.TimeoutExpired:
        error = f"Script timed out after {timeout} seconds"
        print(f"[script_runner] ⏰ {error}")
        return False, [error], error

    except Exception as e:
        error = f"Unexpected error running script: {str(e)}"
        print(f"[script_runner] ❌ {error}")
        return False, [error], error