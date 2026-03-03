import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from agent.graph import build_graph


def run_agent(pdf_path: str, max_fix_attempts: int = None):
    """
    Main function to run the research paper → notebook agent.

    Args:
        pdf_path: Path to the research paper PDF
        max_fix_attempts: Max number of fix attempts (default: from .env or 3)
    """
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.lower().endswith(".pdf"):
        print(f"Error: File must be a PDF: {pdf_path}")
        sys.exit(1)

    if max_fix_attempts is None:
        max_fix_attempts = int(os.getenv("MAX_FIX_ATTEMPTS", "3"))

    print("\n" + "=" * 60)
    print("Research Paper → Notebook Agent")
    print("=" * 60)
    print(f"PDF:               {pdf_path}")
    print(f"Max fix attempts:  {max_fix_attempts}")
    print(f"Model:             {os.getenv('LLM_MODEL', 'qwen3-coder:480b-cloud')}")
    print("=" * 60 + "\n")

    graph = build_graph()
 
    initial_state = {
        "pdf_path":            pdf_path,
        "raw_text":            "",
        "paper_info":          {},
        "script_code":         "",
        "script_path":         "",
        "notebook_path":       "",
        "execution_output":    "",
        "errors":              [],
        "fix_attempts":        0,
        "max_fix_attempts":    max_fix_attempts,
        "is_valid":            False,
        "final_script_path":   "",
        "final_notebook_path": "",
        "error_report":        None,
    }

    final_state = graph.invoke(initial_state)
 
    print("\n" + "=" * 60)
    print("AGENT COMPLETE")
    print("=" * 60)
    print(f"Script:   {final_state['final_script_path']}")
    print(f"Notebook: {final_state['final_notebook_path']}")

    if final_state.get("is_valid"):
        print("Status: SUCCESS — script runs clean!")
    else:
        print(f"Status: PARTIAL — {final_state['fix_attempts']} fix attempts made, errors remain")
        if final_state.get("error_report"):
            print(f"Error report: {final_state['error_report']}")

    print("=" * 60 + "\n")
    return final_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a research paper PDF into a working Jupyter notebook"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the research paper PDF"
    )
    parser.add_argument(
        "--max-fix-attempts",
        type=int,
        default=None,
        help="Max fix attempts (default: from .env or 3)"
    )

    args = parser.parse_args()
    run_agent(args.pdf_path, args.max_fix_attempts)