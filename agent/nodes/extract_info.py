import json
import os
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from agent.state import AgentState


def extract_info_node(state: AgentState) -> AgentState:
    """
    Use LLM to extract structured information from raw paper text.
    """
    print("[extract_info] Extracting structured info from paper using LLM...")

    prompt_template = Path("prompts/extract_info.txt").read_text()
 
    raw_text = state["raw_text"]
    if len(raw_text) > 80000:
        raw_text = raw_text[:80000] + "\n\n[...truncated for length...]"

    prompt = prompt_template.replace("{raw_text}", raw_text)

    llm = ChatOllama(
        model=os.getenv("LLM_MODEL", "qwen3-coder:480b-cloud"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()
 
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        paper_info = json.loads(response_text)
        print(f"[extract_info] Successfully extracted info for: {paper_info.get('title', 'Unknown Title')}")
    except json.JSONDecodeError as e:
        print(f"[extract_info] Warning: Could not parse JSON response: {e}")
        paper_info = {"raw_extraction": response_text}

    return {
        **state,
        "paper_info": paper_info
    }