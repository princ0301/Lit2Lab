import os

from agent.state import AgentState
from tools.rich_ui import ask_execution_approval
from tools.hardware_check import get_hardware_info


def hitl_execution_node(state: AgentState) -> AgentState:
    """
    Show script preview + hardware info,
    ask user to approve execution or skip.
    """
    script_code = state.get("script_code", "")
    default_timeout = int(os.getenv("EXECUTION_TIMEOUT", "300"))

    hw_info = get_hardware_info()

    approved, timeout = ask_execution_approval(script_code, hw_info, default_timeout)

    if not approved:
        print("[hitl_execution] User skipped execution")
    else:
        print(f"[hitl_execution] User approved execution (timeout: {timeout}s)")

    return {
        **state,
        "skip_execution": not approved,
        "execution_timeout": timeout,
    }