import re
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

def py_to_notebook(script_path: str, notebook_path: str, kernel_name: str = "python3") -> str:
    script = Path(script_path).read_text(encoding="utf-8")
    lines = script.splitlines()

    cells = []
    current_block = []
    current_type = None

    def flush(block, btype):
        if not block:
            return
        content = "\n".join(block).strip()
        if not content:
            return
        if btype == "markdown":
            md_lines = []
            for l in block:
                stripped = l.strip()
                if stripped.startswith("# "):
                    md_lines.append(stripped[2:])
                elif stripped == "#":
                    md_lines.append("")
                else:
                    md_lines.append(stripped)
            cells.append(new_markdown_cell("\n".join(md_lines).strip()))
        else:
            cells.append(new_code_cell(content))

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("# ##") or stripped.startswith("# ---") or stripped.startswith("# =="):
            flush(current_block, current_type)
            current_block = [line]
            current_type = "markdown"
            continue

        is_comment = stripped.startswith("#") and not stripped.startswith("#!")

        if is_comment:
            if current_type == "markdown":
                current_block.append(line)
            else:
                flush(current_block, current_type)
                current_block = [line]
                current_type = "markdown"
        else:
            if current_type == "code":
                current_block.append(line)
            else:
                flush(current_block, current_type)
                current_block = [line]
                current_type = "code"

    flush(current_block, current_type)

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": f"Python ({kernel_name})",
        "language": "python",
        "name": kernel_name
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.10.0"
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"[py_to_notebook] Converted {script_path} → {notebook_path}")
    return notebook_path