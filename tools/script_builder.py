from pathlib import Path

def save_script(script_code: str, path: str) -> str:
    Path(path).write_text(script_code, encoding="utf-8")
    return path

def load_script(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")