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
        "  [green][1][/green] Original dataset — real data, best results\n"
        "  [yellow][2][/yellow] Sample / subset — small portion, faster\n"
        "  [blue][3][/blue] Dummy / synthetic — fake data, just test code\n"
        "  [magenta][4][/magenta] Custom path — I have it downloaded locally",
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
 
def ask_web_search_approval(planned_queries: list) -> tuple:
    """
    Show planned search queries and ask user approval.
    Returns (approved: bool, extra_terms: str)
    """
    console.print()
    console.print(Rule("[bold cyan] Web Search[/bold cyan]"))

    console.print("[bold]Planned search queries:[/bold]")
    for i, q in enumerate(planned_queries, 1):
        console.print(f"  [cyan]{i}.[/cyan] {q}")
    console.print()

    approved = Confirm.ask(
        "[bold]Search the web for implementations and resources?[/bold]",
        default=True
    )

    extra_terms = ""
    if approved:
        extra_terms = Prompt.ask(
            "[bold]Any specific terms to add to search?[/bold] [dim](press Enter to skip)[/dim]",
            default=""
        )

    return approved, extra_terms
 
def show_search_results(results: list) -> None:
    """Display web search results in a table."""
    if not results:
        console.print("[yellow]No web search results found.[/yellow]")
        return

    console.print()
    console.print(Rule("[bold cyan] Web Search Results[/bold cyan]"))

    table = Table(box=box.ROUNDED, show_header=True, padding=(0, 1))
    table.add_column("#", style="bold yellow", width=3)
    table.add_column("Title", style="bold white", width=35)
    table.add_column("URL", style="cyan", width=40)
    table.add_column("Snippet", style="white", width=50)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r.get("title", "")[:35],
            r.get("url", "")[:40],
            r.get("content", "")[:80] + "...",
        )

    console.print(table)
    console.print()
 
def ask_execution_approval(script_code: str, hw_info: dict, timeout: int) -> tuple:
    """
    Show script preview and hardware info, ask execution approval.
    Returns (approved: bool, timeout: int)
    """
    console.print()
    console.print(Rule("[bold cyan] Execution Review[/bold cyan]"))
 
    gpu_status = (
        f"[green] {hw_info['gpu_name']}[/green]"
        if hw_info.get("has_gpu")
        else "[yellow] No GPU detected — training may be slow[/yellow]"
    )

    hw_text = (
        f"  CPU: {hw_info.get('cpu', 'Unknown')}\n"
        f"  RAM: {hw_info.get('ram_gb', '?')} GB\n"
        f"  GPU:  {gpu_status}\n"
        f"  Timeout: {timeout}s ({timeout//60}m)"
    )
    console.print(Panel(hw_text, title="[bold cyan]Hardware[/bold cyan]", box=box.ROUNDED))
 
    preview_lines = script_code.splitlines()[:40]
    preview = "\n".join(preview_lines)
    if len(script_code.splitlines()) > 40:
        preview += f"\n\n... ({len(script_code.splitlines()) - 40} more lines)"

    console.print()
    console.print(Panel(
        Syntax(preview, "python", theme="monokai", line_numbers=True),
        title="[bold cyan]Script Preview (first 40 lines)[/bold cyan]",
        box=box.ROUNDED,
    ))

    console.print()
    console.print(Panel(
        "  [green][1][/green] Run it now\n"
        "  [yellow][2][/yellow] Change timeout and run\n"
        "  [red][3][/red] Skip execution (just save script + notebook)",
        title="[bold cyan]Execution Options[/bold cyan]",
        box=box.ROUNDED,
    ))

    choice = Prompt.ask("[bold]Your choice[/bold]", choices=["1", "2", "3"], default="1")

    if choice == "1":
        return True, timeout
    elif choice == "2":
        new_timeout = int(Prompt.ask("[bold]Set timeout in seconds[/bold]", default=str(timeout)))
        return True, new_timeout
    else:
        return False, timeout
 
def ask_post_execution(execution_output: str, script_path: str) -> str:
    """
    Show execution results and ask what to do next.
    Returns: "finish" / "rerun" / "tweak"
    """
    console.print()
    console.print(Rule("[bold green] Execution Complete[/bold green]"))
 
    output_lines = execution_output.strip().splitlines()
    tail = "\n".join(output_lines[-30:])
    console.print(Panel(
        tail or "[dim]No output[/dim]",
        title="[bold]Execution Output (last 30 lines)[/bold]",
        box=box.ROUNDED
    ))

    console.print()
    console.print(Panel(
        "  [green][1][/green] Convert to notebook and finish \n"
        "  [yellow][2][/yellow] Re-run with different dataset choice\n"
        "  [blue][3][/blue] Tweak hyperparameters and re-run",
        title="[bold cyan]What next?[/bold cyan]",
        box=box.ROUNDED,
    ))

    choice = Prompt.ask("[bold]Your choice[/bold]", choices=["1", "2", "3"], default="1")
    mapping = {"1": "finish", "2": "rerun", "3": "tweak"}
    return mapping[choice]
 
def ask_error_review(errors: list, fix_attempts: int, max_attempts: int) -> str:
    """
    Show errors and ask how to proceed.
    Returns: "autofix" / "abort"
    """
    console.print()
    console.print(Rule("[bold red] Execution Failed[/bold red]"))

    for i, err in enumerate(errors, 1):
        console.print(Panel(
            err[:600],
            title=f"[bold red]Error {i}[/bold red]",
            box=box.ROUNDED,
            border_style="red"
        ))

    console.print()
    attempts_left = max_attempts - fix_attempts
    console.print(f"[dim]Fix attempts used: {fix_attempts}/{max_attempts} "
                  f"({attempts_left} remaining)[/dim]")
    console.print()

    console.print(Panel(
        f"  [green][1][/green] Auto-fix and retry  [dim]({attempts_left} attempts left)[/dim]\n"
        "  [red][2][/red] Abort — save as-is with error report",
        title="[bold cyan]How to proceed?[/bold cyan]",
        box=box.ROUNDED,
    ))

    choice = Prompt.ask("[bold]Your choice[/bold]", choices=["1", "2"], default="1")
    return "autofix" if choice == "1" else "abort"
 
def show_session_summary(state: dict, elapsed_seconds: float) -> None:
    """Show final session summary table."""
    from tools.error_memory import get_memory_stats

    console.print()
    console.print(Rule("[bold green] Session Complete[/bold green]"))

    mins = int(elapsed_seconds // 60)
    secs = int(elapsed_seconds % 60)

    stats = get_memory_stats()

    table = Table(box=box.DOUBLE_EDGE, show_header=False, padding=(0, 2))
    table.add_column("Field", style="bold cyan",  width=22)
    table.add_column("Value", style="bold white", width=40)

    paper_title = state.get("paper_info", {}).get("title", "Unknown")[:40]
    is_valid = state.get("is_valid", False)
    fix_attempts = state.get("fix_attempts", 0)
    dataset = state.get("dataset_choice", "dummy")
    script_path = state.get("final_script_path", "N/A")
    nb_path = state.get("final_notebook_path", "N/A")

    table.add_row("Paper", paper_title)
    table.add_row("Script Status", "[green]Clean[/green]" if is_valid else "[red]Errors remain[/red]")
    table.add_row("Dataset", dataset)
    table.add_row("Fix Rounds", str(fix_attempts))
    table.add_row("Time Taken", f"{mins}m {secs}s")
    table.add_row("Memory Total", f"{stats['total_entries']} entries, {stats['successful_fixes']} successful")
    table.add_row("Script", script_path)
    table.add_row("Notebook",     nb_path)

    console.print(table)
    console.print()
 
    if nb_path and nb_path != "N/A":
        open_nb = Confirm.ask("[bold]Open notebook in VS Code now?[/bold]", default=False)
        if open_nb:
            import subprocess
            subprocess.Popen(["code", nb_path])
 
def spinner(message: str):
    """Returns a Rich status spinner context manager."""
    return console.status(f"[bold cyan]{message}[/bold cyan]", spinner="dots")