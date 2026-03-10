from agent.state import AgentState
from tools.rich_ui import show_dataset_options

def hitl_dataset_node(state: AgentState) -> AgentState:
    """
    Show dataset info, let user choose
    original / sample / dummy / custom path.
    """
    paper_info = state.get("paper_info", {})
    datasets = paper_info.get("datasets", [])

    dataset_choice, custom_path = show_dataset_options(datasets)

    print(f"[hitl_dataset] User chose: {dataset_choice}"
          + (f" → {custom_path}" if custom_path else ""))

    return {
        **state,
        "dataset_choice": dataset_choice,
        "custom_dataset_path": custom_path,
    }
