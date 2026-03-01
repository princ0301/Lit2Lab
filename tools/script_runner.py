import subprocess
from typing import Tuple
from tools.kernel_detector import get_agent_python


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
            print(f"[script_runner] Script exited cleanly (code 0)")
            return True, [], full_output
        else:
            print(f"[script_runner] Script exited with code {result.returncode}")
            errors = [stderr] if stderr else [f"Exit code {result.returncode} with no stderr"]
            return False, errors, full_output

    except subprocess.TimeoutExpired:
        error = f"Script timed out after {timeout} seconds"
        print(f"[script_runner] {error}")
        return False, [error], error

    except Exception as e:
        error = f"Unexpected error running script: {str(e)}"
        print(f"[script_runner] {error}")
        return False, [error], error