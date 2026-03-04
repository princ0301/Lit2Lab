import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

MEMORY_DIR = Path("memory")
MEMORY_FILE = MEMORY_DIR / "error_memory.json"
MAX_MEMORY_ENTRIES = 500

def _load_memory() -> list:
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return []
    
def _save_memory(entries: list) -> None:
    MEMORY_DIR.mkdir(exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

def _extract_error_type(error: str) -> str:
    for line in reversed(error.strip().splitlines()):
        line = line.strip()
        if line and ":" in line:
            candidate = line.split(":")[0].strip()
            if candidate and candidate[0].isupper() and " " not in candidate:
                return candidate
            
    return "UnknownError"

def _extract_error_signature(error: str) -> str:
    lines = error.strip().splitlines()
    sig_lines = [l.strip() for l in lines[-3:] if l.strip()]
    return " | ".join(sig_lines)[:300]

def _similarity_score(query_sig: str, stored_sig: str) -> float:
    query_tokens = set(query_sig.lower().split())
    stored_tokens = set(stored_sig.lower().split())
    if not query_tokens or not stored_tokens:
        return 0.0
    intersection = query_tokens & stored_tokens
    union = query_tokens | stored_tokens
    return len(intersection) / len(union)

def store_fix(
    error: str,
    fixed_code_snippet: str,
    fix_summary: str,
    was_successful: bool,
    paper_title: str = "unknown",
    fix_attempt_number: int = 1,
):
    entries = _load_memory()

    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "error_signature": _extract_error_signature(error),
        "error_type": _extract_error_type(error),
        "fix_summary": fix_summary,
        "fixed_code_snippet": fixed_code_snippet[:500],   
        "was_successful": was_successful,
        "paper_title": paper_title,
        "fix_attempt_number": fix_attempt_number,
    }

    entries.append(entry)

    if len(entries) > MAX_MEMORY_ENTRIES:
        entries = entries[-MAX_MEMORY_ENTRIES:]

    _save_memory(entries)

    status = "Successful" if was_successful else "Failed"
    print(f"[error_memory] Stored fix {status}: {entry['error_type']}")

def retrieve_similar_fixex(error: str, top_k: int = 5) -> list:
    entries = _load_memory()
    successful = [e for e in entries if e.get("was_successful", False)]

    if not successful:
        return []
    
    query_sig = _extract_error_signature(error)

    scored = []
    for entry in successful:
        score = _similarity_score(query_sig, entry["error_signature"])
        if score > 0.1:
            scored.append((score, entry))

    scored.sort(key=lambda x: (x[0], x[1]["timestamp"]), reverse=True)

    return [entry for _, entry in scored[:top_k]]

def format_fixex_for_prompt(similar_fixes: list) -> str:
    if not similar_fixes:
        return ""
    
    lines = [
        "PAST SUCCESSFUL FIXES (learn from these):",
        "=" * 50,
    ]

    for i, fix in enumerate(similar_fixes, 1):
        lines += [
            f"\n[Example {i}]",
            f"Error Type:   {fix['error_type']}",
            f"Error:        {fix['error_signature']}",
            f"How it was fixed: {fix['fix_summary']}",
            f"Code snippet: {fix['fixed_code_snippet']}",
        ]

    lines.append("=" * 50)
    return "\n".join(lines)

def get_memory_stats() -> dict:
    entries = _load_memory()
    successful = [e for e in entries if e.get("was_successful", False)]
    failed = [e for e in entries if not e.get("was_successful", False)]

    error_types = {}
    for e in entries:
        et = e.get("error_type", "Unknown")
        error_types[et] = error_types.get(et, 0) + 1

    return {
        "total_entries": len(entries),
        "successful_fixes": len(successful),
        "failed_fixes": len(failed),
        "top_error_types": sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5],
    }