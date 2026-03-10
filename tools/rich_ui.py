from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.text import Text
from rich import box
from rich.rule import Rule

console = Console()

def show_paper_summary(paper_info: dict) -> None:
    """Display a formatted summary of the extracted paper info."""
    console.print()
    console.print(Rule("[bold cyan] Paper Summary[/bold cyan]"))

    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
    table.add_column("Field", style="bold cyan", width=18)
    table.add_column("Value", style="white")

    table.add_row("Title", paper_info.get("title", "Unknown"))
    table.add_row("Objective", paper_info.get("objective", "N/A"))
    table.add_row("Methods", ", ".join(paper_info.get("methods", [])))
    table.add_row("Metrics", ", ".join(paper_info.get("evaluation_metrics", [])))
    table.add_row("Dependencies", ", ".join(paper_info.get("dependencies", [])))

    datasets = paper_info.get("datasets", [])
    if datasets:
        ds_text = "\n".join(
            f"• {d.get('name', '?')} — {d.get('description', '')[:60]}"
            for d in datasets
        )
        table.add_row("Datasets", ds_text)

    console.print(table)
    console.print()