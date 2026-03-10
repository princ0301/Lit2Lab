import os
from typing import Optional

def _search_tavily(query: str, max_results: int = 5) -> list:
    """Search using Tavily API."""
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_raw_content=False
    )
    return response.get("results", [])

def _search_duckduckgo(query: str, max_results: int = 5) -> list:
    """Fallback: search using DuckDuckGo"""
    from duckduckgo_search import DDGS
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("href", ""),
                "content": r.get("body", ""),
            })
    return results

def search(query: str, max_results: int = 5) -> list:
    """
    Search the web. Uses Tavily if TAVILY_API_KEY is set, else DuckDuckGo.
    Returns list of {title, url, content} dicts.
    """
    has_tavily = bool(os.getenv("TAVILY_API_KEY", "").strip())

    try:
        if has_tavily:
            return _search_tavily(query, max_results)
        else:
            print("[tavily_search] No TAVILY_API_KEY found → using DuckDuckGo")
            return _search_duckduckgo(query, max_results)
    except Exception as e:
        print(f"[tavily_search] Search failed: {e}")
        return []
    
def build_search_queries(paper_info: dict, extra_terms: str = "") -> list:
    """
    Build a list of targeted search queries from paper info.
    """
    title = paper_info.get("title", "")
    methods = paper_info.get("methods", [])
    datasets = [d.get("name", "") for d in paper_info.get("datasets", [])]

    queries = []

    if title:
        queries.append(f"{title} github implementation code")
    
    if methods:
        queries.append(f"{methods[0]} python implementation tutorial")

    if datasets and datasets[0]:
        queries.append(f"{datasets[0]} dataset download python")
 
    if title:
        queries.append(f"{title} implementation common errors fixes")


    if extra_terms.strip():
        queries.append(extra_terms.strip())

    return queries[:5] 

def format_results_for_prompt(results: list) -> str:
    """Format search results into a string for LLM prompt injection."""
    if not results:
        return ""
    
    lines = [
        "WEB SEARCH RESULTS (use these to write better code):",
        "=" * 50,
    ]

    for i, r in enumerate(results, 1):
        lines += [
            f"\n[Result {i}]",
            f"Title: {r.get('title', '')}",
            f"URL: {r.get('url', '')}",
            f"Content: {r.get('content', '')[:400]}",
        ]

    lines.append("=" * 50)
    return "\n".join(lines)