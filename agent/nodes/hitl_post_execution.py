from agent.state import AgentState
from tools.rich_ui import ask_post_execution


def hitl_post_execution_node(state: AgentState) -> AgentState:
    """
    Show execution results, ask what to do next.
    Options: finish / rerun with different dataset / tweak hyperparams
    """
    execution_output = state.get("execution_output", "")
    script_path = state.get("script_path", "")

    action = ask_post_execution(execution_output, script_path)

    rerun_requested = action in ("rerun", "tweak")

    if rerun_requested:
        print(f"[hitl_post_execution] User requested: {action}")
    else:
        print("[hitl_post_execution] User chose to finish")

    return {
        **state,
        "rerun_requested": rerun_requested,
    }