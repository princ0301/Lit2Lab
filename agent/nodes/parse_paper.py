from agent.state import AgentState
from tools.pdf_parser import parse_pdf

def parse_paper_node(state: AgentState) -> AgentState:
    """
    Parse the PDF and extract raw text.
    """
    print(f"[parse_paper] Parsing PDF: {state['pdf_path']}")
    raw_text = parse_pdf(state["pdf_path"])
    print(f"[parse_paper] Extracted {len(raw_text)} characters from PDF")

    return {
        **state,
        "raw_text": raw_text
    }