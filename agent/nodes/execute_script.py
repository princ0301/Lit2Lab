from agent.state import AgentState
from tools.script_runner import run_script
from tools.rich_ui import spinner


def execute_script_node(state: AgentState) -> AgentState:
    """
    Execute the Python script.
    Respects skip_execution flag and user-set timeout.
    """
    if state.get("skip_execution", False):
        print("[execute_script] Skipped by user")
        return {
            **state,
            "is_valid": False,
            "errors": ["Execution skipped by user"],
            "execution_output": "",
        }

    script_path = state["script_path"]
    timeout = state.get("execution_timeout", 300)

    print(f"[execute_script] Running: {script_path} (timeout: {timeout}s)")

    with spinner(f"Running script..."):
        is_valid, errors, execution_output = run_script(script_path, timeout=timeout)

    if is_valid:
        print("[execute_script] Script executed successfully!")
    else:
        print(f"[execute_script] Script failed with {len(errors)} error(s)")
        if errors:
            print(f"  → {str(errors[0])[:300]}")

    return {
        **state,
        "is_valid": is_valid,
        "errors": errors,
        "execution_output": execution_output,
    }