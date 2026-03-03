import os
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import json

from agent.state import AgentState
from tools.script_builder import save_script

def generate_script_node(state: AgentState) -> AgentState:
    """
    Use LLM to generate a complete script from paper info.
    """
    print(f"[generate_script] Generating Python script from paper info...")

    prompt_template = Path("prompts/generate_script.txt").read_text()
    prompt = prompt_template.replace("{paper_info}", json.dumps(state["paper_info"], indent=2))

    llm = ChatOllama(
        model=os.getenv("LLM_MODEL", "qwen3-coder:480b-cloud"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    script_code = response.content.strip()

    if script_code.startswith("```"):
        lines = script_code.split("\n")
        script_code = "\n".join(lines[1:-1]).strip()

    title = state["paper_info"].get("title", "script").replace(" ", "_")[:50]
    title = "".join(c for c in title if c.isalnum() or c in "_-")
    script_path = f"outputs/{title}.py"
    save_script(script_code, script_path)

    print(f"[generate_script] Script saved to: {script_path} ({len(script_code.splitlines())} lines)")

    return {
        **state,
        "script_code": script_code,
        "script_path": script_path,
    }