from pathlib import Path

from agent.state import AgentState
from tools.py_to_notebook import py_to_notebook
from tools.kernel_detector import detect_best_kernel

def save_output_node(state: AgentState) -> AgentState:
    """
    Finalize output.
    - Always saves the .py script (already on disk)
    - Converts .py → .ipynb for the user
    - Writes error report if still failing after max retries
    """

    script_path = state["script_path"]
    notebook_path = script_path.replace(".py", ".ipynb")
    error_report = None

    try:
        kernel = detect_best_kernel()
        py_to_notebook(script_path, notebook_path, kernel_name=kernel)
        print(f"[save_output] Notebook saved to: {notebook_path}")
    except Exception as e:
        print(f"[save_output] Could not convert to notebook: {e}")
        notebook_path = ""

    if not state.get("is_valid", False):
        report_path = script_path.replace(".py", "_error_report.txt")
        report_lines = [
            f"Script:       {script_path}",
            f"Fix attempts: {state.get('fix_attempts', 0)} / {state['max_fix_attempts']}",
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

        report_text = "\n".join(report_lines)
        Path(report_path).write_text(report_text, encoding="utf-8")
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