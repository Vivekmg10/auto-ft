import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="autofinetune",
    help="Autonomous LLM finetuning agent",
    add_completion=False,
)
console = Console()


@app.command()
def init(
    config: str = typer.Option(..., "--config", "-c", help="Path to experiment YAML"),
    output_dir: str = typer.Option("./experiments", "--output", "-o", help="Output directory"),
):
    """Initialize a new experiment from a config file."""
    from autofinetune.config.loader import load_config
    from autofinetune.storage.local import LocalStorage

    cfg = load_config(config)
    storage = LocalStorage(output_dir)
    storage.init_experiment(cfg.id)

    # save a copy of the config into the experiment dir for reporter to use later
    import yaml, shutil
    exp_dir = storage.experiment_dir(cfg.id)
    shutil.copy(config, exp_dir / "config_used.yaml")

    rprint(f"[green]✓[/green] Experiment initialized: [bold]{cfg.id}[/bold]")
    rprint(f"  Output: {output_dir}/{cfg.id}")
    rprint(f"  Model: {cfg.base_model.name}")
    rprint(f"  Mode: {cfg.training.mode}")
    rprint(f"  Max runs: {cfg.training.max_runs}")


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to experiment YAML"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume existing experiment"),
):
    """Start or resume an experiment."""
    from autofinetune.config.loader import load_config
    from autofinetune.storage.local import LocalStorage
    from autofinetune.graph.builder import build_graph, create_initial_state
    from autofinetune.agents.strategist import StrategistAgent
    from autofinetune.agents.data_agent import DataAgent
    from autofinetune.agents.monitor import TrainingMonitor
    from autofinetune.agents.evaluator import EvaluatorAgent
    from autofinetune.memory.compressor import MemoryCompressor
    from autofinetune.journal.writer import JournalWriter

    cfg = load_config(config)
    storage = LocalStorage(cfg.storage.base_path)

    # resume or start fresh
    if resume:
        state = storage.load_state(cfg.id)
        if not state:
            rprint("[yellow]No existing state found — starting fresh[/yellow]")
            state = create_initial_state(cfg)
    else:
        state = create_initial_state(cfg)

    storage.init_experiment(cfg.id)

    # save config copy for reporter
    import shutil
    exp_dir = storage.experiment_dir(cfg.id)
    shutil.copy(config, exp_dir / "config_used.yaml")

    rprint(f"\n[bold green]AutoFineTune[/bold green]")
    rprint(f"Experiment: [bold]{cfg.id}[/bold]")
    rprint(f"Model: {cfg.base_model.name}")
    rprint(f"Max runs: {cfg.training.max_runs}\n")

    # wire up agents
    strategist = StrategistAgent(model=cfg.agents.strategist)
    data_agent = DataAgent(model=cfg.agents.data_agent)
    monitor = TrainingMonitor(model=cfg.agents.monitor)
    evaluator = EvaluatorAgent(model=cfg.agents.evaluator)
    memory = MemoryCompressor(model=cfg.agents.strategist)
    journal = JournalWriter(model=cfg.agents.strategist)

    # build graph
    graph = build_graph(
        config=cfg,
        strategist=strategist,
        data_agent=data_agent,
        monitor=monitor,
        evaluator=evaluator,
        memory=memory,
        journal=journal,
        storage=storage,
    )

    thread_config = {"configurable": {"thread_id": cfg.id}}

    try:
        for event in graph.stream(
            state.model_dump(),
            config=thread_config,
            stream_mode="updates",
        ):
            node_name = list(event.keys())[0]
            rprint(f"[dim]→ {node_name}[/dim]")

            # save state after every node
            updated = event[node_name]
            if updated:
                current_state = storage.load_state(cfg.id) or state
                merged = current_state.model_copy(update=updated)
                storage.save_state(merged)

    except KeyboardInterrupt:
        rprint("\n[yellow]Experiment paused. Run with --resume to continue.[/yellow]")
    except Exception as e:
        rprint(f"\n[red]Experiment failed: {e}[/red]")
        raise


@app.command()
def status(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    output_dir: str = typer.Option("./experiments", "--output", "-o"),
):
    """Show current status and leaderboard for an experiment."""
    from autofinetune.storage.local import LocalStorage
    from autofinetune.memory.run_db import RunDatabase

    storage = LocalStorage(output_dir)
    state = storage.load_state(experiment_id)

    if not state:
        rprint(f"[red]Experiment not found: {experiment_id}[/red]")
        raise typer.Exit(1)

    rprint(f"\n[bold]{experiment_id}[/bold]")
    rprint(f"Phase: [bold]{state.current_phase}[/bold]")
    rprint(f"Progress: {state.runs_completed}/{state.max_runs} runs")
    rprint(f"Hours used: {state.total_hours_used:.1f}h")
    rprint(f"Best score: {state.best_score:.4f}" if state.best_score else "Best score: N/A")

    db = RunDatabase(storage.experiment_dir(experiment_id))
    leaderboard = db.get_leaderboard()

    if leaderboard:
        rprint()
        table = Table(title="Leaderboard", show_header=True)
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Run ID", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("LR")
        table.add_column("LoRA Rank")
        table.add_column("Epochs")
        table.add_column("Scheduler")

        for entry in leaderboard[:10]:
            cfg_data = entry.get("config", {})
            table.add_row(
                str(entry["rank"]),
                entry["run_id"],
                f"{entry['eval_score']:.4f}",
                str(cfg_data.get("learning_rate", "")),
                str(cfg_data.get("lora_rank", "")),
                str(cfg_data.get("epochs", "")),
                str(cfg_data.get("scheduler", "")),
            )

        console.print(table)


@app.command()
def report(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    output_dir: str = typer.Option("./experiments", "--output", "-o"),
    regenerate: bool = typer.Option(
        False, "--regenerate", "-g",
        help="Regenerate report even if one already exists"
    ),
):
    """Generate or display the final experiment report."""
    from autofinetune.storage.local import LocalStorage
    from autofinetune.journal.reporter import ExperimentReporter
    from autofinetune.config.loader import load_config

    storage = LocalStorage(output_dir)
    report_path = storage.experiment_dir(experiment_id) / "report.md"

    # serve existing report unless --regenerate
    if report_path.exists() and not regenerate:
        rprint(report_path.read_text())
        return

    # load config to get model + use case
    config_path = storage.experiment_dir(experiment_id) / "config_used.yaml"

    if config_path.exists():
        cfg = load_config(str(config_path))
        reporter_model = cfg.agents.strategist
        use_case = cfg.use_case
    else:
        rprint("[yellow]config_used.yaml not found — using fallback model[/yellow]")
        reporter_model = "anthropic/claude-opus-4-5"
        use_case = "LLM finetuning experiment"

    rprint(f"[dim]Generating report for {experiment_id}...[/dim]")

    reporter = ExperimentReporter(model=reporter_model)
    report_text = reporter.generate(
        experiment_dir=storage.experiment_dir(experiment_id),
        experiment_id=experiment_id,
        use_case=use_case,
    )

    storage.save_report(experiment_id, report_text)
    rprint(report_text)


@app.command()
def dashboard(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    output_dir: str = typer.Option("./experiments", "--output", "-o"),
):
    """Open the live TUI dashboard for a running experiment."""
    from autofinetune.cli.dashboard import AutoFineTuneDashboard

    dash = AutoFineTuneDashboard(
        experiment_id=experiment_id,
        output_dir=output_dir,
    )
    dash.run()


@app.command()
def logs(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    run_id: str = typer.Option(None, "--run", "-r", help="Specific run ID"),
    output_dir: str = typer.Option("./experiments", "--output", "-o"),
):
    """Show journal entries for an experiment."""
    from autofinetune.storage.local import LocalStorage

    storage = LocalStorage(output_dir)
    journal_dir = storage.experiment_dir(experiment_id) / "journal"

    if not journal_dir.exists():
        rprint(f"[yellow]No journal entries yet for {experiment_id}[/yellow]")
        return

    if run_id:
        entry_path = journal_dir / f"{run_id}.md"
        if entry_path.exists():
            rprint(entry_path.read_text())
        else:
            rprint(f"[red]No journal entry for run {run_id}[/red]")
    else:
        entries = sorted(journal_dir.glob("run_*.md"))
        for entry in entries:
            rprint(entry.read_text())
            rprint("\n" + "─" * 60 + "\n")


if __name__ == "__main__":
    app()