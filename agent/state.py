from typing import TypedDict, Optional

class AgentState(TypedDict):
    pdf_path: str
    raw_text: str
    paper_info: dict
    script_code: str
    script_path: str
    notebook_path: str
    execution_output: str
    errors: list
    fix_attempts: int
    max_fix_attempts: int
    is_valid: bool
    final_notebook_path: str
    final_script_path: str
    error_report: Optional[str]