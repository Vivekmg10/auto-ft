from pathlib import Path
from datetime import datetime
import json
import shutil
from autofinetune.graph.state import ExperimentState, RunResult


class LocalStorage:
    def __init__(self, base_path: str):
        self.base = Path(base_path)

    def experiment_dir(self, experiment_id: str) -> Path:
        return self.base / experiment_id

    def run_dir(self, experiment_id: str, run_id: str) -> Path:
        return self.experiment_dir(experiment_id) / "runs" / run_id

    def init_experiment(self, experiment_id: str):
        dirs = [
            self.experiment_dir(experiment_id),
            self.experiment_dir(experiment_id) / "runs",
            self.experiment_dir(experiment_id) / "journal",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def save_state(self, state: ExperimentState):
        path = self.experiment_dir(state.experiment_id) / "state.json"
        path.write_text(state.model_dump_json(indent=2))

    def load_state(self, experiment_id: str) -> ExperimentState | None:
        path = self.experiment_dir(experiment_id) / "state.json"
        if not path.exists():
            return None
        return ExperimentState.model_validate_json(path.read_text())

    def save_run(self, experiment_id: str, run: RunResult):
        run_path = self.run_dir(experiment_id, run.run_id)
        run_path.mkdir(parents=True, exist_ok=True)

        (run_path / "config.json").write_text(
            json.dumps(run.config.model_dump(), indent=2)
        )
        (run_path / "results.json").write_text(
            json.dumps({
                "run_id": run.run_id,
                "eval_score": run.eval_score,
                "eval_breakdown": run.eval_breakdown,
                "status": run.status,
                "training_report": run.training_report
            }, indent=2)
        )
        (run_path / "hypothesis.json").write_text(
            json.dumps({"hypothesis": run.hypothesis}, indent=2)
        )
        (run_path / "loss_curve.json").write_text(
            json.dumps(run.loss_curve, indent=2)
        )

    def update_leaderboard(self, experiment_id: str, state: ExperimentState):
        completed = [r for r in state.all_runs if r.status == "completed"]
        ranked = sorted(completed, key=lambda r: r.eval_score or 0, reverse=True)

        leaderboard = [
            {
                "rank": i + 1,
                "run_id": r.run_id,
                "eval_score": r.eval_score,
                "config": r.config.model_dump(),
                "hypothesis": r.hypothesis[:100] + "..." if len(r.hypothesis) > 100 else r.hypothesis
            }
            for i, r in enumerate(ranked)
        ]

        path = self.experiment_dir(experiment_id) / "leaderboard.json"
        path.write_text(json.dumps(leaderboard, indent=2))

    def save_journal_entry(self, experiment_id: str, run_id: str, entry: str):
        path = self.experiment_dir(experiment_id) / "journal" / f"{run_id}.md"
        path.write_text(entry)

    def save_summary(self, experiment_id: str, summary: str):
        path = self.experiment_dir(experiment_id) / "journal" / "summary.md"
        path.write_text(summary)

    def save_report(self, experiment_id: str, report: str):
        path = self.experiment_dir(experiment_id) / "report.md"
        path.write_text(report)

    def checkpoint_dir(self, experiment_id: str, run_id: str) -> Path:
        return self.run_dir(experiment_id, run_id) / "checkpoint"

    def cleanup_checkpoints(self, experiment_id: str, keep_run_ids: list[str]):
        runs_dir = self.experiment_dir(experiment_id) / "runs"
        if not runs_dir.exists():
            return
        for run_dir in runs_dir.iterdir():
            if run_dir.name not in keep_run_ids:
                checkpoint = run_dir / "checkpoint"
                if checkpoint.exists():
                    shutil.rmtree(checkpoint)