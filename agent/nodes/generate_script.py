import os
import json
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from agent.state import AgentState
from tools.script_builder import save_script
from tools.rich_ui import spinner


def generate_script_node(state: AgentState) -> AgentState:
    """
    Generate complete Python script from paper info,
    web search context, and user's dataset choice.
    """
    print("[generate_script] Generating Python script from paper info...")

    prompt_template = Path("prompts/generate_script.txt").read_text()
 
    dataset_choice = state.get("dataset_choice", "dummy")
    custom_path = state.get("custom_dataset_path", "")

    dataset_instruction = {
        "original": "Use the ORIGINAL dataset from the paper. Include download code. Handle download failures gracefully with a synthetic fallback.",
        "sample":   "Use a SMALL SAMPLE or subset of the original dataset (max 1000 rows / 10% of data). Include download code with subset slicing.",
        "dummy":    "Use a SYNTHETIC/DUMMY dataset. Do NOT attempt to download anything. Generate fake data using numpy/sklearn make_* functions.",
        "custom":   f"The user has provided a local dataset at: {custom_path}. Load data from this path. Handle file not found with a synthetic fallback.",
    }.get(dataset_choice, "Use a synthetic dummy dataset.")
 
    prompt = (
        prompt_template
        .replace("{paper_info}", json.dumps(state["paper_info"], indent=2))
        .replace("{web_context}", state.get("web_context", "No web search results available."))
        .replace("{dataset_instruction}", dataset_instruction)
    )

    llm = ChatOllama(
        model=os.getenv("LLM_MODEL", "qwen3-coder:480b-cloud"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    )

    with spinner("Generating script..."):
        response = llm.invoke([HumanMessage(content=prompt)])

    script_code = response.content.strip()
 
    if script_code.startswith("```"):
        lines = script_code.split("\n")
        script_code = "\n".join(lines[1:-1]).strip()
 
    title = state["paper_info"].get("title", "script").replace(" ", "_")[:50]
    title = "".join(c for c in title if c.isalnum() or c in "_-")
    script_path = f"outputs/{title}.py"
    save_script(script_code, script_path)

    print(f"[generate_script] Script saved: {script_path} ({len(script_code.splitlines())} lines)")

    return {
        **state,
        "script_code": script_code,
        "script_path": script_path,
    }