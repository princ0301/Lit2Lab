from agent.state import AgentState
from tools.rich_ui import ask_error_review


def hitl_error_review_node(state: AgentState) -> AgentState:
    """
    On script failure, show errors and ask user
    whether to auto-fix or abort.
    """
    errors = state.get("errors", [])
    fix_attempts = state.get("fix_attempts", 0)
    max_attempts = state.get("max_fix_attempts", 3)

    action = ask_error_review(errors, fix_attempts, max_attempts)

    if action == "abort":
        print("[hitl_error_review] User chose to abort")
    else:
        print("[hitl_error_review] User approved auto-fix")

    return {
        **state,
        "abort": action == "abort",
    }