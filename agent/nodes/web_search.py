from agent.state import AgentState
from tools.tavily_search import build_search_queries, search, format_results_for_prompt
from tools.rich_ui import show_search_results, spinner

def web_search_node(state: AgentState) -> AgentState:
    """
    Web Search Node: Run Tavily/DuckDuckGo queries and collect results.
    Skipped if user declined in HITL checkpoint 1.
    """
    if state.get("skip_web_search", False):
        print("[web_search] Skipped by user")
        return {
            **state,
            "web_search_results": [],
            "web_context": "",
        }
    
    paper_info = state.get("paper_info", {})
    extra_terms = state.get("user_search_terms", "")

    queries = build_search_queries(paper_info, extra_terms)

    all_results = []
    for query in queries:
        with spinner(f"Searching: {query}"):
            results = search(query, max_results=3)
            all_results.extend(results)

    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    unique_results = unique_results[:10]
    show_search_results(unique_results)
    
    web_context = format_results_for_prompt(unique_results)

    print(f"[web_search] Found {len(unique_results)} results across {len(queries)} queries")

    return {
        **state,
        "web_search_results": unique_results,
        "web_context": web_context,
    }