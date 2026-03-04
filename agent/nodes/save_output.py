from pathlib import Path

from agent.state import AgentState
from tools.py_to_notebook import py_to_notebook
from tools.kernel_detector import detect_best_kernel
from tools.error_memory import store_fix, get_memory_stats

def save_output_node(state: AgentState) -> AgentState:
    """
    Finalize output.
    - Converts .py → .ipynb
    - If script is valid and fixes were made → stores successful fix in memory
    - If still failing → writes error report
    """
    script_path = state["script_path"]
    notebook_path = script_path.replace(".py", ".ipynb")
    error_report = None
    paper_title = state.get("paper_info", {}).get("title", "unknown")
 
    try:
        kernel = detect_best_kernel()
        py_to_notebook(script_path, notebook_path, kernel_name=kernel)
        print(f"[save_output] Notebook saved to: {notebook_path}")
    except Exception as e:
        print(f"[save_output] Could not convert to notebook: {e}")
        notebook_path = ""
 
    is_valid = state.get("is_valid", False)
    fix_attempts = state.get("fix_attempts", 0)

    if is_valid and fix_attempts > 0:
        errors_str = "\n".join(state.get("errors", []))
        store_fix(
            error=errors_str,
            fixed_code_snippet=state.get("script_code", "")[:500],
            fix_summary=f"Fixed after {fix_attempts} attempt(s) — script ran successfully",
            was_successful=True,
            paper_title=paper_title,
            fix_attempt_number=fix_attempts,
        )
        print(f"[save_output] Successful fix stored in memory!")
 
    stats = get_memory_stats()
    print(f"[save_output] Memory: {stats['total_entries']} total entries, "
          f"{stats['successful_fixes']} successful fixes stored")
 
    if not is_valid:
        report_path = script_path.replace(".py", "_error_report.txt")
        report_lines = [
            f"Script:       {script_path}",
            f"Fix attempts: {fix_attempts} / {state['max_fix_attempts']}",
            f"Status:       FAILED - unresolved errors remain\n",
            "=" * 60,
            "ERRORS:",
            "=" * 60,
        ]
        for i, err in enumerate(state.get("errors", []), 1):
            report_lines.append(f"\n--- Error {i} ---\n{err}")

        report_lines += [
            "\n" + "=" * 60,
            "LAST EXECUTION OUTPUT:",
            "=" * 60,
            state.get("execution_output", "N/A"),
        ]

        Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")
        print(f"[save_output] Error report saved to: {report_path}")
        error_report = report_path
    else:
        print(f"[save_output] Script is clean and valid!")

    return {
        **state,
        "final_script_path": script_path,
        "final_notebook_path": notebook_path,
        "error_report": error_report,
    }