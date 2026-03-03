from agent.state import AgentState
from tools.script_runner import run_script

def execute_script(state: AgentState) -> AgentState:
    """
    Execute the Python script ans capture errors.
    """
    print(f"[execute_script] Running script: {state["script_path"]}")

    is_valid, errors, execution_output = run_script(state["script_path"])

    if is_valid:
        print("[execute_script] Script executed successfully with no errors!")
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