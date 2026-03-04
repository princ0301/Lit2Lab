import os
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from agent.state import AgentState
from tools.script_builder import save_script
from tools.error_memory import retrieve_similar_fixes, format_fixes_for_prompt, store_fix

def _extract_fix_summary(original: str, fixed: str) -> str:
    """Generate a short summary of what changed between original and fixed script."""
    original_lines = set(original.splitlines())
    fixed_lines = set(fixed.splitlines())
    added = [l.strip() for l in (fixed_lines - original_lines) if l.strip()][:5]
    if added:
        return "Changes: " + " | ".join(added)[:200]
    return "Script was rewritten"


def fix_script_node(state: AgentState) -> AgentState:
    """
    Use LLM to fix errors in the Python script.
    Retrieves similar past fixes from memory before prompting,
    and stores the result after each attempt.
    """
    fix_attempts = state.get("fix_attempts", 0) + 1
    print(f"[fix_script] Fix attempt {fix_attempts}/{state['max_fix_attempts']}...")

    errors_str = "\n".join(state["errors"])
    paper_title = state.get("paper_info", {}).get("title", "unknown")
 
    similar_fixes = retrieve_similar_fixes(errors_str, top_k=5)
    memory_context = format_fixes_for_prompt(similar_fixes)
    if similar_fixes:
        print(f"[fix_script] Injecting {len(similar_fixes)} similar past fix(es) from memory")
    else:
        print(f"[fix_script] No similar past fixes found yet (memory is building up)")
 
    prompt_template = Path("prompts/fix_script.txt").read_text()
    prompt = (
        prompt_template
        .replace("{script_code}", state["script_code"])
        .replace("{errors}", errors_str)
        .replace("{execution_output}", state.get("execution_output", "N/A"))
        .replace("{fix_attempts}", str(fix_attempts))
        .replace("{max_fix_attempts}", str(state["max_fix_attempts"]))
        .replace("{memory_context}", memory_context)
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
        store_fix(
            error=errors_str,
            fixed_code_snippet="",
            fix_summary="LLM returned empty response",
            was_successful=False,
            paper_title=paper_title,
            fix_attempt_number=fix_attempts,
        )
        return {**state, "fix_attempts": fix_attempts}
 
    save_script(fixed_code, state["script_path"])
    print(f"[fix_script] Fixed script saved to: {state['script_path']}")
 
    fix_summary = _extract_fix_summary(state["script_code"], fixed_code)
    store_fix(
        error=errors_str,
        fixed_code_snippet=fixed_code[:500],
        fix_summary=fix_summary,
        was_successful=False,    
        paper_title=paper_title,
        fix_attempt_number=fix_attempts,
    )

    return {
        **state,
        "script_code": fixed_code,
        "fix_attempts": fix_attempts,
    }