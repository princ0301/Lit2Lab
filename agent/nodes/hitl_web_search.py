from agent.state import AgentState
from tools.rich_ui import show_paper_summary, ask_web_search_approval
from tools.tavily_search import build_search_queries

def hitl_web_search_node(state: AgentState) -> AgentState:
    """
    Show paper summary, ask web search approval.
    """
    paper_info = state.get("paper_info", {})
    show_paper_summary(paper_info)
    planned_queries = build_search_queries(paper_info)
    approved, extra_terms = ask_web_search_approval(planned_queries)

    return {
        **state,
        "skip_web_search": not approved,
        "user_search_terms": extra_terms,
    }
