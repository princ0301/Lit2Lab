from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.parse_paper import parse_paper_node
from agent.nodes.extract_info import extract_info_node
from agent.nodes.hitl_web_search import hitl_web_search_node
from agent.nodes.web_search import web_search_node
from agent.nodes.hitl_dataset import hitl_dataset_node
from agent.nodes.generate_script import generate_script_node
from agent.nodes.hitl_execution import hitl_execution_node
from agent.nodes.execute_script import execute_script_node
from agent.nodes.hitl_error_review import hitl_error_review_node
from agent.nodes.fix_script import fix_script_node
from agent.nodes.hitl_post_execution import hitl_post_execution_node
from agent.nodes.save_output import save_output_node
 
def after_hitl_error_review(state: AgentState) -> str:
    """Abort or auto-fix based on user choice."""
    if state.get("abort", False):
        return "save_output"
    return "fix_script"


def after_execute_script(state: AgentState) -> str:
    """
    After execution:
    - skip_execution → save directly
    - valid → post execution HITL
    - invalid + attempts left → error review HITL
    - invalid + max attempts → save with error report
    """
    if state.get("skip_execution", False):
        return "save_output"

    if state.get("is_valid", False):
        return "hitl_post_execution"

    fix_attempts = state.get("fix_attempts", 0)
    max_attempts = state.get("max_fix_attempts", 3)

    if fix_attempts >= max_attempts:
        return "save_output"

    return "hitl_error_review"


def after_hitl_post_execution(state: AgentState) -> str:
    """After post execution: finish or rerun."""
    if state.get("rerun_requested", False):
        return "hitl_dataset"   # go back to dataset selection
    return "save_output"
 
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
 
    graph.add_node("parse_paper",           parse_paper_node)
    graph.add_node("extract_info",          extract_info_node)
    graph.add_node("hitl_web_search",       hitl_web_search_node)
    graph.add_node("web_search",            web_search_node)
    graph.add_node("hitl_dataset",          hitl_dataset_node)
    graph.add_node("generate_script",       generate_script_node)
    graph.add_node("hitl_execution",        hitl_execution_node)
    graph.add_node("execute_script",        execute_script_node)
    graph.add_node("hitl_error_review",     hitl_error_review_node)
    graph.add_node("fix_script",            fix_script_node)
    graph.add_node("hitl_post_execution",   hitl_post_execution_node)
    graph.add_node("save_output",           save_output_node)
 
    graph.set_entry_point("parse_paper")
    graph.add_edge("parse_paper",         "extract_info")
    graph.add_edge("extract_info",        "hitl_web_search")
    graph.add_edge("hitl_web_search",     "web_search")
    graph.add_edge("web_search",          "hitl_dataset")
    graph.add_edge("hitl_dataset",        "generate_script")
    graph.add_edge("generate_script",     "hitl_execution")
    graph.add_edge("hitl_execution",      "execute_script")

    # After execution: valid → post HITL, invalid → error review HITL or save
    graph.add_conditional_edges(
        "execute_script",
        after_execute_script,
        {
            "hitl_post_execution": "hitl_post_execution",
            "hitl_error_review":   "hitl_error_review",
            "save_output":         "save_output",
        }
    )

    # Error review: fix or abort
    graph.add_conditional_edges(
        "hitl_error_review",
        after_hitl_error_review,
        {
            "fix_script":  "fix_script",
            "save_output": "save_output",
        }
    )

    # Fix → re-execute
    graph.add_edge("fix_script", "execute_script")

    # Post execution: finish or rerun from dataset selection
    graph.add_conditional_edges(
        "hitl_post_execution",
        after_hitl_post_execution,
        {
            "hitl_dataset": "hitl_dataset",
            "save_output":  "save_output",
        }
    )

    graph.add_edge("save_output", END)

    return graph.compile()