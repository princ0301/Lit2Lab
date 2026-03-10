from typing import TypedDict, Optional

class AgentState(TypedDict):
    pdf_path: str

    raw_text: str

    paper_info: dict

    web_search_results: list
    web_context: str 

    user_search_terms: str
    skip_web_search: bool
    dataset_choice: str
    custom_dataset_path: str
    skip_execution: bool
    execution_timeout: int
    abort: bool

    script_code: str
    script_path: str

    execution_output: str
    errors: list

    fix_attempts: int
    max_fix_attempts: int
    is_valid: bool

    rerun_requested: bool 

    notebook_path: str
    final_notebook_path: str
    final_script_path: str
    error_report: Optional[str]