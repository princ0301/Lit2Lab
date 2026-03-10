import os
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()


def run_agent(pdf_path: str, max_fix_attempts: int = None):
    if not Path(pdf_path).exists():
        console.print(f"[red] PDF not found: {pdf_path}[/red]")
        sys.exit(1)

    if not pdf_path.lower().endswith(".pdf"):
        console.print(f"[red] File must be a PDF: {pdf_path}[/red]")
        sys.exit(1)

    if max_fix_attempts is None:
        max_fix_attempts = int(os.getenv("MAX_FIX_ATTEMPTS", "3"))

    console.print()
    console.print(Panel(
        f"  [cyan]PDF:[/cyan]              {pdf_path}\n"
        f"  [cyan]Max fix attempts:[/cyan] {max_fix_attempts}\n"
        f"  [cyan]Model:[/cyan]            {os.getenv('LLM_MODEL', 'qwen3-coder:480b-cloud')}",
        title="[bold cyan] Research Paper → Notebook Agent[/bold cyan]",
        box=box.DOUBLE_EDGE,
    ))

    from agent.graph import build_graph
    graph = build_graph()

    initial_state = {
        "pdf_path":             pdf_path,
        "raw_text":             "",
        "paper_info":           {},
        "web_search_results":   [],
        "web_context":          "",
        "user_search_terms":    "",
        "skip_web_search":      False,
        "dataset_choice":       "dummy",
        "custom_dataset_path":  "",
        "skip_execution":       False,
        "execution_timeout":    int(os.getenv("EXECUTION_TIMEOUT", "300")),
        "abort":                False,
        "script_code":          "",
        "script_path":          "",
        "execution_output":     "",
        "errors":               [],
        "fix_attempts":         0,
        "max_fix_attempts":     max_fix_attempts,
        "is_valid":             False,
        "rerun_requested":      False,
        "notebook_path":        "",
        "final_script_path":    "",
        "final_notebook_path":  "",
        "error_report":         None,
    }

    start = time.time()
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - start
 
    from tools.rich_ui import show_session_summary
    show_session_summary(final_state, elapsed)

    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a research paper PDF into a working Jupyter notebook"
    )
    parser.add_argument("pdf_path", type=str, help="Path to the research paper PDF")
    parser.add_argument("--max-fix-attempts", type=int, default=None)

    args = parser.parse_args()
    run_agent(args.pdf_path, args.max_fix_attempts)