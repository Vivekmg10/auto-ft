import json
from pathlib import Path
from loguru import logger
from autofinetune.graph.state import RunResult, ExperimentState


class RunDatabase:
    """
    Manages run history as JSON files on disk.
    Acts as the query layer over the runs/ directory.
    """

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.runs_dir = experiment_dir / "runs"
        self.leaderboard_path = experiment_dir / "leaderboard.json"

    def get_all_runs(self) -> list[RunResult]:
        """Load all runs from disk."""
        runs = []
        if not self.runs_dir.exists():
            return runs

        for run_dir in sorted(self.runs_dir.iterdir()):
            run = self._load_run(run_dir)
            if run:
                runs.append(run)

        return runs

    def get_run(self, run_id: str) -> RunResult | None:
        run_dir = self.runs_dir / run_id
        return self._load_run(run_dir)

    def get_best_run(self) -> RunResult | None:
        runs = self.get_all_runs()
        completed = [r for r in runs if r.status == "completed" and r.eval_score]
        if not completed:
            return None
        return max(completed, key=lambda r: r.eval_score)

    def get_leaderboard(self) -> list[dict]:
        if not self.leaderboard_path.exists():
            return []
        return json.loads(self.leaderboard_path.read_text())

    def query(
        self,
        status: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        priority: str | None = None,
    ) -> list[RunResult]:
        """
        Simple query interface over run history.
        Strategist uses this to find specific runs.
        """
        runs = self.get_all_runs()

        if status:
            runs = [r for r in runs if r.status == status]
        if min_score is not None:
            runs = [r for r in runs if r.eval_score and r.eval_score >= min_score]
        if max_score is not None:
            runs = [r for r in runs if r.eval_score and r.eval_score <= max_score]

        return runs

    def get_history_for_strategist(self) -> list[dict]:
        """
        Returns a compact summary of all runs suitable for the strategist prompt.
        """
        runs = self.get_all_runs()
        summary = []

        for run in runs:
            summary.append({
                "run_id": run.run_id,
                "status": run.status,
                "hypothesis": run.hypothesis,
                "config": run.config.model_dump() if run.config else {},
                "eval_score": run.eval_score,
                "eval_breakdown": run.eval_breakdown,
                "failure_reason": run.failure_reason,
            })

        return summary

    # ── private ───────────────────────────────────────────────────────────────

    def _load_run(self, run_dir: Path) -> RunResult | None:
        if not run_dir.is_dir():
            return None

        try:
            config_path = run_dir / "config.json"
            results_path = run_dir / "results.json"
            hypothesis_path = run_dir / "hypothesis.json"
            loss_curve_path = run_dir / "loss_curve.json"

            if not results_path.exists():
                return None

            results = json.loads(results_path.read_text())

            config = {}
            if config_path.exists():
                config = json.loads(config_path.read_text())

            hypothesis = ""
            if hypothesis_path.exists():
                hypothesis = json.loads(hypothesis_path.read_text()).get("hypothesis", "")

            loss_curve = []
            if loss_curve_path.exists():
                loss_curve = json.loads(loss_curve_path.read_text())

            from autofinetune.graph.state import RunConfig
            return RunResult(
                run_id=results["run_id"],
                config=RunConfig(**config) if config else RunConfig(learning_rate=2e-4),
                hypothesis=hypothesis,
                eval_score=results.get("eval_score"),
                eval_breakdown=results.get("eval_breakdown", {}),
                best_checkpoint_path=results.get("best_checkpoint_path"),
                loss_curve=loss_curve,
                training_report=results.get("training_report", {}),
                status=results.get("status", "unknown"),
                failure_reason=results.get("failure_reason"),
            )

        except Exception as e:
            logger.warning(f"Failed to load run from {run_dir}: {e}")
            return None 