import json
from pathlib import Path
from datetime import datetime
from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    Footer,
    Static,
    DataTable,
    ProgressBar,
    Label,
    RichLog,
)
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual import work
import asyncio


class ExperimentInfo(Static):
    """Top panel showing experiment overview."""

    def compose(self) -> ComposeResult:
        yield Label("", id="exp-title")
        yield Label("", id="exp-phase")
        yield Label("", id="exp-progress")
        yield Label("", id="exp-best")
        yield Label("", id="exp-hours")


class CurrentRun(Static):
    """Middle panel showing current run details."""

    def compose(self) -> ComposeResult:
        yield Label("Current Run", classes="panel-title")
        yield Label("", id="run-id")
        yield Label("", id="run-hypothesis")
        yield ProgressBar(total=100, id="run-progress", show_eta=False)
        yield Label("", id="run-loss")


class Leaderboard(Static):
    """Right panel showing ranked runs."""

    def compose(self) -> ComposeResult:
        yield Label("Leaderboard", classes="panel-title")
        yield DataTable(id="leaderboard-table")

    def on_mount(self):
        table = self.query_one("#leaderboard-table", DataTable)
        table.add_columns("Rank", "Run ID", "Score", "LR", "Rank", "Status")


class JournalPanel(Static):
    """Bottom panel showing latest journal entry."""

    def compose(self) -> ComposeResult:
        yield Label("Research Journal", classes="panel-title")
        yield RichLog(id="journal-log", wrap=True, markup=True)


class AutoFineTuneDashboard(App):
    """
    Live TUI dashboard for monitoring AutoFineTune experiments.
    Polls experiment state every 5 seconds and updates all panels.
    """

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2;
        grid-rows: 1fr 1fr 1fr;
    }

    ExperimentInfo {
        height: 100%;
        border: solid $primary;
        padding: 1 2;
    }

    CurrentRun {
        height: 100%;
        border: solid $accent;
        padding: 1 2;
    }

    Leaderboard {
        height: 100%;
        column-span: 2;
        border: solid $success;
        padding: 1 2;
    }

    JournalPanel {
        height: 100%;
        column-span: 2;
        border: solid $warning;
        padding: 1 2;
    }

    .panel-title {
        text-style: bold;
        color: $text-muted;
        margin-bottom: 1;
    }

    #exp-title {
        text-style: bold;
        font-size: 16;
    }

    ProgressBar {
        margin: 1 0;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self, experiment_id: str, output_dir: str):
        super().__init__()
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.experiment_dir = self.output_dir / experiment_id

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield ExperimentInfo()
        yield CurrentRun()
        yield Leaderboard()
        yield JournalPanel()
        yield Footer()

    def on_mount(self):
        self.title = f"AutoFineTune — {self.experiment_id}"
        self.refresh_data()
        # poll every 5 seconds
        self.set_interval(5, self.refresh_data)

    def refresh_data(self):
        state = self._load_state()
        if state:
            self._update_experiment_info(state)
            self._update_current_run(state)

        leaderboard = self._load_leaderboard()
        if leaderboard:
            self._update_leaderboard(leaderboard)

        latest_journal = self._load_latest_journal()
        if latest_journal:
            self._update_journal(latest_journal)

    def _update_experiment_info(self, state: dict):
        self.query_one("#exp-title", Label).update(
            f"[bold]{state.get('experiment_id', '')}[/bold]"
        )
        self.query_one("#exp-phase", Label).update(
            f"Phase: {state.get('current_phase', '').upper()}"
        )
        runs_done = state.get('runs_completed', 0)
        max_runs = state.get('max_runs', 0)
        self.query_one("#exp-progress", Label).update(
            f"Progress: {runs_done}/{max_runs} runs"
        )
        best = state.get('best_score', 0)
        best_id = state.get('best_run_id', 'None')
        self.query_one("#exp-best", Label).update(
            f"Best: {best:.4f} ({best_id})" if best else "Best: N/A"
        )
        hours = state.get('total_hours_used', 0)
        self.query_one("#exp-hours", Label).update(
            f"Hours used: {hours:.1f}h"
        )

    def _update_current_run(self, state: dict):
        run_id = state.get('current_run_id') or "Waiting..."
        hypothesis = state.get('current_hypothesis') or "—"
        phase = state.get('current_phase', '')

        self.query_one("#run-id", Label).update(f"Run: {run_id}")
        self.query_one("#run-hypothesis", Label).update(
            f"[dim]{hypothesis[:120]}...[/dim]"
            if len(hypothesis) > 120 else f"[dim]{hypothesis}[/dim]"
        )

        # rough progress based on phase
        phase_progress = {
            "init": 0,
            "data_prep": 10,
            "planning": 20,
            "training": 60,
            "evaluating": 85,
            "update": 95,
            "complete": 100,
        }
        progress = phase_progress.get(phase, 0)
        self.query_one("#run-progress", ProgressBar).update(progress=progress)

        # show latest loss if in training
        all_runs = state.get("all_runs", [])
        current_losses = []
        for run in all_runs:
            if run.get("run_id") == state.get("current_run_id"):
                current_losses = run.get("loss_curve", [])

        if current_losses:
            latest = current_losses[-1]
            self.query_one("#run-loss", Label).update(
                f"Step {latest.get('step', '?')} | "
                f"train_loss={latest.get('train_loss', 'N/A')} | "
                f"eval_loss={latest.get('eval_loss', 'N/A')}"
            )

    def _update_leaderboard(self, leaderboard: list[dict]):
        table = self.query_one("#leaderboard-table", DataTable)
        table.clear()

        for entry in leaderboard[:8]:
            cfg = entry.get("config", {})
            table.add_row(
                str(entry.get("rank", "")),
                entry.get("run_id", ""),
                f"{entry.get('eval_score', 0):.4f}",
                str(cfg.get("learning_rate", "")),
                str(cfg.get("lora_rank", "")),
                "✓",
            )

    def _update_journal(self, entry: str):
        log = self.query_one("#journal-log", RichLog)
        log.clear()
        # show last 30 lines of latest journal entry
        lines = entry.split("\n")[-30:]
        log.write("\n".join(lines))

    def _load_state(self) -> dict | None:
        state_path = self.experiment_dir / "state.json"
        if not state_path.exists():
            return None
        try:
            return json.loads(state_path.read_text())
        except Exception:
            return None

    def _load_leaderboard(self) -> list[dict] | None:
        lb_path = self.experiment_dir / "leaderboard.json"
        if not lb_path.exists():
            return None
        try:
            return json.loads(lb_path.read_text())
        except Exception:
            return None

    def _load_latest_journal(self) -> str | None:
        journal_dir = self.experiment_dir / "journal"
        if not journal_dir.exists():
            return None
        entries = sorted(journal_dir.glob("run_*.md"))
        if not entries:
            return None
        try:
            return entries[-1].read_text()
        except Exception:
            return None

    def action_refresh(self):
        self.refresh_data()