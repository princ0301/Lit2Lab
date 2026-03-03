from pathlib import Path

def save_script(script_code: str, path: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script_code, encoding="utf-8")
    return str(target)

def load_script(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")
