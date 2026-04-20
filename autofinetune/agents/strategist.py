import json
import re
from loguru import logger
from autofinetune.agents.base import BaseAgent


STRATEGIST_PROMPT = """
You are an expert ML researcher running a hyperparameter search experiment.
Your job is to study the history of previous finetuning runs and propose
the single most promising next experiment to run.

You think carefully and scientifically. You:
- Identify patterns in what has worked and what hasn't
- Form clear hypotheses before proposing configs
- Balance exploration (trying new things) vs exploitation (refining what works)
- Reason about WHY a config might work, not just what to try
- Consider the budget — if few runs remain, exploit more than explore

You always respond in this exact format:

THINKING:
<your step by step reasoning about the run history and what to try next>

HYPOTHESIS:
<one clear sentence explaining what you're testing and why you expect it to help>

CONFIG:
```json
{
  "learning_rate": <float>,
  "lora_rank": <int>,
  "lora_alpha": <int>,
  "lora_dropout": <float>,
  "batch_size": <int>,
  "gradient_accumulation": <int>,
  "warmup_ratio": <float>,
  "weight_decay": <float>,
  "epochs": <int>,
  "scheduler": <string>,
  "max_grad_norm": <float>
}
```

PRIORITY:
<"explore" or "exploit">

JOURNAL:
<2-3 sentences written in first person for the research journal, 
summarizing your reasoning and what you hope to learn from this run>
"""


class StrategistAgent(BaseAgent):
    def __init__(self, model: str):
        super().__init__(
            model=model,
            system_prompt=STRATEGIST_PROMPT,
            tools=[],               # strategist only reasons, no tool calls
            temperature=0.8,        # slightly higher for creative exploration
            max_iterations=1,       # one shot — no tool loop needed
        )

    def _build_prompt(self, task: str, context: dict) -> str:
        parts = []

        parts.append(f"GOAL:\n{context['use_case']}\n")

        parts.append(
            f"EXPERIMENT PROGRESS:\n"
            f"Run {context['current_run_number']} of {context['max_runs']} | "
            f"{context['budget_remaining']} runs remaining | "
            f"{context['hours_used']:.1f}h used\n"
        )

        parts.append(f"TRAINING MODE: {context['training_mode']}\n")

        if context.get("best_run"):
            best = context["best_run"]
            parts.append(
                f"CURRENT BEST:\n"
                f"Run: {best['run_id']} | Score: {best['score']:.4f}\n"
                f"Config: {json.dumps(best['config'], indent=2)}\n"
                f"Hypothesis: {best['hypothesis']}\n"
            )
        else:
            parts.append("CURRENT BEST: None yet — this is the first run\n")

        if context.get("lessons_learned"):
            lessons = "\n".join(f"- {l}" for l in context["lessons_learned"])
            parts.append(f"LESSONS LEARNED SO FAR:\n{lessons}\n")

        parts.append(f"RUN HISTORY:\n{context['run_history']}\n")

        if context.get("hp_space"):
            parts.append(
                f"HYPERPARAMETER SEARCH SPACE:\n"
                f"{json.dumps(context['hp_space'], indent=2)}\n"
            )

        parts.append(f"\nTASK: {task}")

        return "\n".join(parts)

    def _parse_output(self, content: str) -> dict:
        """
        Parses the structured output from the strategist.
        Extracts hypothesis, config JSON, priority, and journal entry.
        """
        try:
            hypothesis = self._extract_section(content, "HYPOTHESIS")
            config_str = self._extract_json(content)
            priority = self._extract_section(content, "PRIORITY").strip().lower()
            journal = self._extract_section(content, "JOURNAL")
            thinking = self._extract_section(content, "THINKING")

            config = json.loads(config_str)

            # validate priority
            if priority not in ("explore", "exploit"):
                priority = "explore"

            return {
                "hypothesis": hypothesis.strip(),
                "config": config,
                "priority": priority,
                "journal_entry": journal.strip(),
                "thinking": thinking.strip(),
            }

        except Exception as e:
            logger.error(f"Failed to parse strategist output: {e}")
            logger.debug(f"Raw output: {content}")
            return self._fallback_config()

    def _extract_section(self, content: str, section: str) -> str:
        """Extract text between a section header and the next section."""
        pattern = rf"{section}:\s*(.*?)(?=\n[A-Z]+:|```json|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_json(self, content: str) -> str:
        """Extract JSON block from the config section."""
        pattern = r"```json\s*(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        # fallback — try to find raw JSON object
        pattern = r"\{[^{}]*\}"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(0)
        raise ValueError("No JSON config found in strategist output")

    def _fallback_config(self) -> dict:
        """
        Safe default config if parsing fails.
        This should rarely trigger but prevents the whole run from crashing.
        """
        logger.warning("Using fallback config due to parse failure")
        return {
            "hypothesis": "Fallback run with safe default config",
            "config": {
                "learning_rate": 2e-4,
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "batch_size": 4,
                "gradient_accumulation": 4,
                "warmup_ratio": 0.05,
                "weight_decay": 0.01,
                "epochs": 3,
                "scheduler": "cosine",
                "max_grad_norm": 1.0,
            },
            "priority": "explore",
            "journal_entry": "Parse error — running safe default config.",
            "thinking": "",
        }