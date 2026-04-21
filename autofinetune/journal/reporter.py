import json
from pathlib import Path
from datetime import datetime
from loguru import logger
from litellm import completion


class ExperimentReporter:
    """
    Generates the final experiment report by reading all
    journal entries, run results, and leaderboard from disk.

    Separate from JournalWriter — writer handles per-run entries,
    reporter handles the final synthesis across all runs.
    """

    def __init__(self, model: str):
        self.model = model

    def generate(
        self,
        experiment_dir: Path,
        experiment_id: str,
        use_case: str,
    ) -> str:
        """
        Main entry point. Reads everything from disk,
        synthesizes into a full markdown report.
        """
        leaderboard = self._load_leaderboard(experiment_dir)
        all_runs = self._load_all_runs(experiment_dir)
        journal_entries = self._load_journal_entries(experiment_dir)
        state = self._load_state(experiment_dir)

        best_run = leaderboard[0] if leaderboard else None
        completed = [r for r in all_runs if r.get("status") == "completed"]
        failed = [r for r in all_runs if r.get("status") == "failed"]
        pruned = [r for r in all_runs if r.get("status") == "pruned"]

        sections = [
            self._header(experiment_id, state, best_run),
            self._executive_summary(
                use_case, all_runs, completed, failed, best_run, state
            ),
            self._experiment_setup(use_case, state, all_runs),
            self._key_findings(leaderboard, all_runs, journal_entries),
            self._hyperparameter_analysis(completed),
            self._best_configuration(best_run),
            self._failure_analysis(failed, pruned),
            self._recommendations(leaderboard, completed, use_case),
            self._full_leaderboard_table(leaderboard),
            self._conclusion(best_run, use_case, state),
        ]

        return "\n\n".join(s for s in sections if s)

    # ── sections ──────────────────────────────────────────────────────────────

    def _header(self, experiment_id: str, state: dict, best_run: dict | None) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        best_score = best_run.get("eval_score", 0) if best_run else 0
        runs_done = state.get("runs_completed", 0) if state else 0
        hours = state.get("total_hours_used", 0) if state else 0

        return f"""# AutoFineTune Experiment Report

| Field | Value |
|-------|-------|
| Experiment | `{experiment_id}` |
| Generated | {timestamp} |
| Best Score | {best_score:.4f} |
| Total Runs | {runs_done} |
| Total Hours | {hours:.1f}h |
"""

    def _executive_summary(
        self,
        use_case: str,
        all_runs: list[dict],
        completed: list[dict],
        failed: list[dict],
        best_run: dict | None,
        state: dict,
    ) -> str:
        if not all_runs:
            return "## Executive Summary\n\nNo runs completed."

        prompt = f"""
Write a 2-3 paragraph executive summary for this LLM finetuning experiment.

USE CASE: {use_case}

STATS:
- Total runs: {len(all_runs)}
- Completed: {len(completed)}
- Failed: {len(failed)}
- Best score: {best_run.get('eval_score', 0):.4f if best_run else 'N/A'}
- Best run: {best_run.get('run_id', 'N/A') if best_run else 'N/A'}
- Hours used: {state.get('total_hours_used', 0):.1f if state else 0}h

BEST CONFIG:
{json.dumps(best_run.get('config', {}), indent=2) if best_run else 'N/A'}

Write clearly and concisely. Lead with the most important finding.
No headers, just paragraphs.
"""

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            body = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Executive summary generation failed: {e}")
            body = (
                f"This experiment ran {len(all_runs)} finetuning trials "
                f"for the task: {use_case}. "
                f"Best score achieved: "
                f"{best_run.get('eval_score', 0):.4f if best_run else 'N/A'}."
            )

        return f"## Executive Summary\n\n{body}"

    def _experiment_setup(
        self,
        use_case: str,
        state: dict,
        all_runs: list[dict],
    ) -> str:
        base_model = "Unknown"
        training_mode = "Unknown"

        if all_runs:
            first_run = all_runs[0]
            cfg = first_run.get("config", {})
            training_mode = state.get("training_mode", "lora") if state else "lora"

        lines = [
            "## Experiment Setup",
            "",
            f"**Use Case**: {use_case}",
            f"**Training Mode**: {training_mode}",
            f"**Total Runs**: {len(all_runs)}",
            "",
            "### Search Space Explored",
            "",
        ]

        if all_runs:
            configs = [r.get("config", {}) for r in all_runs]
            lrs = sorted(set(c.get("learning_rate") for c in configs if c.get("learning_rate")))
            ranks = sorted(set(c.get("lora_rank") for c in configs if c.get("lora_rank")))
            schedulers = sorted(set(c.get("scheduler") for c in configs if c.get("scheduler")))
            epochs = sorted(set(c.get("epochs") for c in configs if c.get("epochs")))

            lines += [
                f"- **Learning rates tried**: {lrs}",
                f"- **LoRA ranks tried**: {ranks}",
                f"- **Schedulers tried**: {schedulers}",
                f"- **Epochs tried**: {epochs}",
            ]

        return "\n".join(lines)

    def _key_findings(
        self,
        leaderboard: list[dict],
        all_runs: list[dict],
        journal_entries: list[str],
    ) -> str:
        if not leaderboard:
            return ""

        prompt = f"""
Extract 5-7 key findings from this finetuning experiment.

Each finding should be:
- Specific and supported by the data
- Actionable for future experiments
- Written as a single clear sentence

LEADERBOARD (top runs):
{json.dumps(leaderboard[:10], indent=2)}

ALL RUN SCORES:
{json.dumps([{
    'run_id': r.get('run_id'),
    'eval_score': r.get('eval_score'),
    'config': r.get('config'),
    'status': r.get('status'),
    'failure_reason': r.get('failure_reason'),
} for r in all_runs], indent=2)}

Return ONLY a JSON array of finding strings.
"""

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()
            import re
            match = re.search(r"\[.*\]", content, re.DOTALL)
            findings = json.loads(match.group(0)) if match else []
        except Exception as e:
            logger.warning(f"Key findings generation failed: {e}")
            findings = [
                f"Best score achieved: {leaderboard[0].get('eval_score', 0):.4f}",
                f"Total of {len(all_runs)} configurations explored",
            ]

        lines = ["## Key Findings", ""]
        for i, finding in enumerate(findings, 1):
            lines.append(f"{i}. {finding}")

        return "\n".join(lines)

    def _hyperparameter_analysis(self, completed: list[dict]) -> str:
        if len(completed) < 3:
            return ""

        lines = ["## Hyperparameter Analysis", ""]

        # group by each hyperparameter and compute average score
        hp_names = ["learning_rate", "lora_rank", "epochs", "scheduler", "warmup_ratio"]

        for hp in hp_names:
            groups: dict = {}
            for run in completed:
                cfg = run.get("config", {})
                val = cfg.get(hp)
                score = run.get("eval_score")
                if val is None or score is None:
                    continue
                key = str(val)
                if key not in groups:
                    groups[key] = []
                groups[key].append(score)

            if len(groups) < 2:
                continue

            lines.append(f"### {hp.replace('_', ' ').title()}")
            lines.append("")
            lines.append("| Value | Avg Score | Runs |")
            lines.append("|-------|-----------|------|")

            sorted_groups = sorted(
                groups.items(),
                key=lambda x: sum(x[1]) / len(x[1]),
                reverse=True,
            )

            for val, scores in sorted_groups:
                avg = sum(scores) / len(scores)
                lines.append(f"| {val} | {avg:.4f} | {len(scores)} |")

            lines.append("")

        return "\n".join(lines)

    def _best_configuration(self, best_run: dict | None) -> str:
        if not best_run:
            return ""

        cfg = best_run.get("config", {})
        score = best_run.get("eval_score", 0)
        run_id = best_run.get("run_id", "")
        hypothesis = best_run.get("hypothesis", "")

        lines = [
            "## Best Configuration",
            "",
            f"**Run**: `{run_id}`",
            f"**Score**: {score:.4f}",
            "",
            f"**Hypothesis**: {hypothesis}",
            "",
            "### Config",
            "",
            "```yaml",
        ]

        for key, val in cfg.items():
            lines.append(f"{key}: {val}")

        lines += [
            "```",
            "",
        ]

        # breakdown if available
        breakdown = best_run.get("eval_breakdown", {})
        if breakdown:
            lines += [
                "### Score Breakdown",
                "",
                "| Dimension | Score |",
                "|-----------|-------|",
            ]
            for dim, score in breakdown.items():
                lines.append(f"| {dim.replace('_', ' ').title()} | {score:.4f} |")

        return "\n".join(lines)

    def _failure_analysis(
        self,
        failed: list[dict],
        pruned: list[dict],
    ) -> str:
        if not failed and not pruned:
            return ""

        lines = [
            "## Failure Analysis",
            "",
            f"**Failed runs**: {len(failed)}",
            f"**Pruned runs (early stopped)**: {len(pruned)}",
            "",
        ]

        if failed:
            lines.append("### Failed Runs")
            lines.append("")
            for run in failed:
                lines.append(
                    f"- `{run.get('run_id')}`: "
                    f"{run.get('failure_reason', 'Unknown error')}"
                )
            lines.append("")

        if pruned:
            lines.append("### Pruned Runs")
            lines.append("")
            for run in pruned:
                cfg = run.get("config", {})
                lines.append(
                    f"- `{run.get('run_id')}`: "
                    f"lr={cfg.get('learning_rate')} "
                    f"rank={cfg.get('lora_rank')} "
                    f"— stopped early"
                )

        return "\n".join(lines)

    def _recommendations(
        self,
        leaderboard: list[dict],
        completed: list[dict],
        use_case: str,
    ) -> str:
        if not leaderboard:
            return ""

        prompt = f"""
Based on these finetuning experiment results, write 4-6 specific recommendations
for someone who wants to continue improving this model.

USE CASE: {use_case}

TOP RUNS:
{json.dumps(leaderboard[:5], indent=2)}

Recommendations should cover:
- What hyperparameters to explore further
- What data improvements might help
- What evaluation dimensions need work
- Any architectural suggestions

Be specific. Reference actual numbers from the results.
Write as a numbered list. No JSON, just plain text.
"""

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            body = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Recommendations generation failed: {e}")
            body = "Continue exploring learning rate and LoRA rank combinations."

        return f"## Recommendations for Further Improvement\n\n{body}"

    def _full_leaderboard_table(self, leaderboard: list[dict]) -> str:
        if not leaderboard:
            return ""

        lines = [
            "## Full Leaderboard",
            "",
            "| Rank | Run ID | Score | LR | LoRA Rank | Epochs | Scheduler |",
            "|------|--------|-------|----|-----------|--------|-----------|",
        ]

        for entry in leaderboard:
            cfg = entry.get("config", {})
            lines.append(
                f"| {entry.get('rank')} "
                f"| `{entry.get('run_id')}` "
                f"| {entry.get('eval_score', 0):.4f} "
                f"| {cfg.get('learning_rate')} "
                f"| {cfg.get('lora_rank')} "
                f"| {cfg.get('epochs')} "
                f"| {cfg.get('scheduler')} |"
            )

        return "\n".join(lines)

    def _conclusion(
        self,
        best_run: dict | None,
        use_case: str,
        state: dict,
    ) -> str:
        if not best_run:
            return ""

        score = best_run.get("eval_score", 0)
        run_id = best_run.get("run_id", "")
        hours = state.get("total_hours_used", 0) if state else 0
        runs = state.get("runs_completed", 0) if state else 0

        prompt = f"""
Write a 1 paragraph conclusion for this finetuning experiment report.

USE CASE: {use_case}
BEST SCORE: {score:.4f}
BEST RUN: {run_id}
TOTAL RUNS: {runs}
TOTAL HOURS: {hours:.1f}h

Summarize what was accomplished, whether the result is good for the use case,
and what the next step should be. Keep it to 3-4 sentences.
"""

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            body = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Conclusion generation failed: {e}")
            body = (
                f"The experiment completed {runs} runs over {hours:.1f} hours, "
                f"achieving a best score of {score:.4f} with run {run_id}."
            )

        return f"## Conclusion\n\n{body}"

    # ── loaders ───────────────────────────────────────────────────────────────

    def _load_leaderboard(self, experiment_dir: Path) -> list[dict]:
        path = experiment_dir / "leaderboard.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text())
        except Exception:
            return []

    def _load_all_runs(self, experiment_dir: Path) -> list[dict]:
        runs = []
        runs_dir = experiment_dir / "runs"
        if not runs_dir.exists():
            return runs

        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            try:
                results_path = run_dir / "results.json"
                config_path = run_dir / "config.json"
                hypothesis_path = run_dir / "hypothesis.json"

                if not results_path.exists():
                    continue

                run = json.loads(results_path.read_text())

                if config_path.exists():
                    run["config"] = json.loads(config_path.read_text())

                if hypothesis_path.exists():
                    run["hypothesis"] = json.loads(
                        hypothesis_path.read_text()
                    ).get("hypothesis", "")

                runs.append(run)
            except Exception as e:
                logger.warning(f"Failed to load run {run_dir.name}: {e}")

        return runs

    def _load_journal_entries(self, experiment_dir: Path) -> list[str]:
        journal_dir = experiment_dir / "journal"
        if not journal_dir.exists():
            return []

        entries = []
        for path in sorted(journal_dir.glob("run_*.md")):
            try:
                entries.append(path.read_text())
            except Exception:
                continue

        return entries

    def _load_state(self, experiment_dir: Path) -> dict | None:
        state_path = experiment_dir / "state.json"
        if not state_path.exists():
            return None
        try:
            return json.loads(state_path.read_text())
        except Exception:
            return None