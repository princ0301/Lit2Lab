from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    parse_paper_node,
    extract_info_node,
    generate_script_node,
    execute_script_node,
    fix_script_node,
    save_output_node
)

def should_fix_or_save(state: AgentState) -> str:
    """
    Conditional edge: decide whether to fix errors or move to save.
    """
    if state.get("is_valid", False):
        print("[router] Script is valid → saving")
        return "save_output"
    
    fix_attempts = state.get("fix_attempts", 0)
    max_attempts = state.get("max_fix_attempts", 3)

    if fix_attempts >= max_attempts:
        print(f"[router]  Max fix attempts ({max_attempts}) reached → saving with error report")
        return "save_output"

    print(f"[router] Errors found (attempt {fix_attempts}/{max_attempts}) → fixing")
    return "fix_script"

def build_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph agent graph.
    """
    graph = StateGraph(AgentState)
 
    graph.add_node("parse_paper",      parse_paper_node)
    graph.add_node("extract_info",     extract_info_node)
    graph.add_node("generate_script",  generate_script_node)
    graph.add_node("execute_script",   execute_script_node)
    graph.add_node("fix_script",       fix_script_node)
    graph.add_node("save_output",      save_output_node)
 
    graph.set_entry_point("parse_paper")
    graph.add_edge("parse_paper",     "extract_info")
    graph.add_edge("extract_info",    "generate_script")
    graph.add_edge("generate_script", "execute_script")
 
    graph.add_conditional_edges(
        "execute_script",
        should_fix_or_save,
        {
            "fix_script":  "fix_script",
            "save_output": "save_output",
        }
    )
 
    graph.add_edge("fix_script", "execute_script")
 
    graph.add_edge("save_output", END)

    return graph.compile()