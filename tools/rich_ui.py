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

def show_dataset_options(datasets: list) -> str:
    """
    Show dataset info and ask user to pick original / sample / dummy / custom.
    Returns the user's choice string.
    """
    console.print()
    console.print(Rule("[bold cyan] Dataset Selection[/bold cyan]"))

    if datasets:
        table = Table(box=box.ROUNDED, show_header=True, padding=(0, 1))
        table.add_column("#", style="bold yellow", width=3)
        table.add_column("Dataset", style="bold white", width=20)
        table.add_column("Description", style="white", width=40)
        table.add_column("Source", style="cyan", width=30)

        for i, ds in enumerate(datasets, 1):
            table.add_row(
                str(i),
                ds.get("name", "Unknown"),
                ds.get("description", "N/A")[:60],
                ds.get("source", "Not specified")[:40],
            )
        console.print(table)
    else:
        console.print("[yellow] No specific datasets found in paper.[/yellow]")

    console.print()
    console.print(Panel(
        "[bold]Choose how to handle data:[/bold]\n\n"
        "  [green][1][/green] Original dataset     — real data, best results\n"
        "  [yellow][2][/yellow] Sample / subset      — small portion, faster\n"
        "  [blue][3][/blue] Dummy / synthetic    — fake data, just test code\n"
        "  [magenta][4][/magenta] Custom path          — I have it downloaded locally",
        title="[bold cyan]Dataset Options[/bold cyan]",
        box=box.ROUNDED,
    ))

    choice = Prompt.ask(
        "[bold]Your choice[/bold]",
        choices=["1", "2", "3", "4"],
        default="3"
    )

    mapping = {"1": "original", "2": "sample", "3": "dummy", "4": "custom"}
    chosen = mapping[choice]

    custom_path = ""
    if chosen == "custom":
        custom_path = Prompt.ask("[bold]Enter path to your dataset[/bold]")

    return chosen, custom_path