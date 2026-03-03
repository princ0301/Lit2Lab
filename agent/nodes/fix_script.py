import os
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from agent.state import AgentState
from tools.script_builder import save_script

def fix_script_node(state: AgentState) -> AgentState:
    """
    Use LLM to fix errors in the Python script.
    """
    fix_attempts = state.get("fix_attempts", 0) + 1
    print(f"[fix_script] Fix attempt {fix_attempts}/{state['max_fix_attempts']}...")

    prompt_template = Path("prompts/fix_script.txt").read_text()
    prompt = (
        prompt_template
        .replace("{script_code}", state["script_code"])
        .replace("{errors}", "\n".join(state["errors"]))
        .replace("{execution_output}", state.get("execution_output", "N/A"))
        .replace("{fix_attempts}", str(fix_attempts))
        .replace("{max_fix_attempts}", str(state["max_fix_attempts"]))
    )

    llm = ChatOllama(
        model=os.getenv("LLM_MODEL", "qwen3-coder:480b-cloud"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    fixed_code = response.content.strip()

    if fixed_code.startswith("```"):
        lines = fixed_code.split("\n")
        fixed_code = "\n".join(lines[1:-1]).strip()

    if not fixed_code:
        print("[fix_script] Warning: LLM returned empty response, keeping original script")
        return {**state, "fix_attempts": fix_attempts}
 
    save_script(fixed_code, state["script_path"])
    print(f"[fix_script] Fixed script saved to: {state['script_path']}")

    return {
        **state,
        "script_code": fixed_code,
        "fix_attempts": fix_attempts,
    }