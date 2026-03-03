import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path


def py_to_notebook(script_path: str, notebook_path: str, kernel_name: str = "python3") -> str:
    """
    Converts a validated .py script into a structured .ipynb notebook.

    Splitting logic:
    - Lines starting with '# ##' / '# ---' / '# ==' → section separator → markdown cell
    - Consecutive comment-only lines → markdown cell
    - Everything else → code cell

    Indentation is fully preserved in code cells.
    Returns the path to the saved notebook.
    """
    script = Path(script_path).read_text(encoding="utf-8")
    lines = script.splitlines()

    cells = []
    current_block = []
    current_type = None   

    def flush(block, btype):
        if not block:
            return

        if btype == "markdown":
            md_lines = []
            for line in block:
                stripped = line.strip()
                if stripped.startswith("# "):
                    md_lines.append(stripped[2:])
                elif stripped == "#":
                    md_lines.append("")
                else:
                    md_lines.append(stripped)
            content = "\n".join(md_lines).strip()
            if content:
                cells.append(new_markdown_cell(content))

        else:   
            while block and not block[0].strip():
                block.pop(0)
            while block and not block[-1].strip():
                block.pop()
            if block:
                content = "\n".join(block)   
                cells.append(new_code_cell(content))

    for line in lines:
        stripped = line.strip()
 
        if stripped.startswith("# ##") or stripped.startswith("# ---") or stripped.startswith("# =="):
            flush(current_block, current_type)
            current_block = [line]
            current_type = "markdown"
            continue
 
        if stripped.startswith("#!"):
            if current_type == "code":
                current_block.append(line)
            else:
                flush(current_block, current_type)
                current_block = [line]
                current_type = "code"
            continue
 
        is_pure_comment = stripped.startswith("#") and not line.startswith(" ") and not line.startswith("\t")

        if is_pure_comment:
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

if __name__ == "__main__":
    py_to_notebook("D:/Project_new/Lit2Lab/outputs/Sequence_to_Sequence_Learning_with_Neural_Networks.py", "D:/Project_new/Lit2Lab/outputs/notebook.ipynb")