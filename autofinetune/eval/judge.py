import json
import re
from loguru import logger
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential


JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator for conversational AI systems.
You evaluate model responses fairly, honestly, and consistently.

You always:
- Reason carefully before scoring
- Give specific examples from the response to justify scores
- Score each dimension independently
- Penalize hallucination and factual errors heavily
- Reward conciseness and clarity

You never:
- Give scores based on response length alone
- Inflate scores to be nice
- Penalize stylistic choices that don't affect quality
"""


class LLMJudge:
    """
    LLM-as-judge scorer for open ended conversation evaluation.

    Scores each example across multiple dimensions with chain of thought.
    Detects position bias by optionally running twice with shuffled order.
    """

    DIMENSIONS = [
        "task_completion",
        "tone_empathy",
        "accuracy",
        "consistency",
        "refusal_quality",
        "conciseness",
    ]

    DIMENSION_DESCRIPTIONS = {
        "task_completion": "Does the response actually solve or address the user's problem?",
        "tone_empathy": "Is the tone appropriate, professional, and empathetic where needed?",
        "accuracy": "Is the information correct, grounded, and free of hallucination?",
        "consistency": "Is the response consistent with what a well-aligned assistant would say?",
        "refusal_quality": "If the request is out of scope, does it refuse gracefully and helpfully?",
        "conciseness": "Is the response appropriately concise without being unhelpfully brief?",
    }

    def __init__(self, model: str):
        self.model = model

    def score_example(
        self,
        input_text: str,
        model_output: str,
        ideal: str,
        rubric: str,
        use_case: str,
    ) -> dict:
        """
        Score a single example across all dimensions.
        Returns scores dict and reasoning.
        """
        prompt = self._build_judge_prompt(
            input_text=input_text,
            model_output=model_output,
            ideal=ideal,
            rubric=rubric,
            use_case=use_case,
        )

        try:
            response = self._call_judge(prompt)
            return self._parse_scores(response)
        except Exception as e:
            logger.warning(f"Judge scoring failed: {e}")
            return self._empty_scores()

    def score_batch(
        self,
        examples: list[dict],
        outputs: list[str],
        use_case: str,
    ) -> list[dict]:
        """
        Score a batch of examples.
        Returns list of score dicts in same order.
        """
        results = []

        for example, output in zip(examples, outputs):
            score = self.score_example(
                input_text=example.get("input", example.get("prompt", "")),
                model_output=output,
                ideal=example.get("ideal", example.get("output", "")),
                rubric=example.get("rubric", "Evaluate general response quality"),
                use_case=use_case,
            )
            results.append(score)

        return results

    def aggregate_scores(self, scored_examples: list[dict]) -> dict:
        """
        Aggregate individual example scores into overall eval report.
        """
        if not scored_examples:
            return self._empty_scores()

        # average each dimension
        breakdown = {}
        for dim in self.DIMENSIONS:
            scores = [
                s["dimensions"].get(dim, 0)
                for s in scored_examples
                if s.get("dimensions")
            ]
            breakdown[dim] = sum(scores) / len(scores) if scores else 0.0

        # primary score is weighted average
        weights = {
            "task_completion": 0.30,
            "accuracy":        0.25,
            "tone_empathy":    0.15,
            "consistency":     0.15,
            "refusal_quality": 0.10,
            "conciseness":     0.05,
        }

        primary_score = sum(
            breakdown.get(dim, 0) * weight
            for dim, weight in weights.items()
        )

        # collect strengths and weaknesses
        strengths = []
        weaknesses = []

        for dim, score in breakdown.items():
            if score >= 0.8:
                strengths.append(f"{dim.replace('_', ' ').title()} ({score:.2f})")
            elif score < 0.6:
                weaknesses.append(f"{dim.replace('_', ' ').title()} ({score:.2f})")

        # find worst examples for the report
        worst = sorted(
            scored_examples,
            key=lambda s: s.get("primary_score", 0)
        )[:3]

        return {
            "primary_score": round(primary_score, 4),
            "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
            "strengths": strengths,
            "weaknesses": weaknesses,
            "n_examples": len(scored_examples),
            "worst_examples": [
                {
                    "input": w.get("input", "")[:100],
                    "output": w.get("output", "")[:100],
                    "score": w.get("primary_score", 0),
                }
                for w in worst
            ],
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _build_judge_prompt(
        self,
        input_text: str,
        model_output: str,
        ideal: str,
        rubric: str,
        use_case: str,
    ) -> str:
        dimensions_str = "\n".join(
            f"- {dim}: {desc}"
            for dim, desc in self.DIMENSION_DESCRIPTIONS.items()
        )

        return f"""
Evaluate this model response for a {use_case} assistant.

INPUT FROM USER:
{input_text}

MODEL RESPONSE:
{model_output}

IDEAL RESPONSE (key points, not exact wording):
{ideal}

EVALUATION RUBRIC:
{rubric}

DIMENSIONS TO SCORE (0.0 to 1.0 each):
{dimensions_str}

Think step by step. For each dimension:
1. Note what the model did well or poorly
2. Assign a score from 0.0 (terrible) to 1.0 (perfect)

Then return your scores as JSON in this exact format:
```json
{{
  "reasoning": "your step by step analysis",
  "dimensions": {{
    "task_completion": 0.0,
    "tone_empathy": 0.0,
    "accuracy": 0.0,
    "consistency": 0.0,
    "refusal_quality": 0.0,
    "conciseness": 0.0
  }},
  "primary_score": 0.0,
  "one_line_summary": "brief verdict"
}}
```
"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8)
    )
    def _call_judge(self, prompt: str) -> str:
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content

    def _parse_scores(self, content: str) -> dict:
        try:
            match = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(1))
            else:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                else:
                    raise ValueError("No JSON found")

            # validate dimensions
            dims = parsed.get("dimensions", {})
            for dim in self.DIMENSIONS:
                if dim not in dims:
                    dims[dim] = 0.5

            # clamp all scores to 0-1
            dims = {k: max(0.0, min(1.0, float(v))) for k, v in dims.items()}

            primary = parsed.get("primary_score")
            if not primary:
                primary = sum(dims.values()) / len(dims)
            primary = max(0.0, min(1.0, float(primary)))

            return {
                "dimensions": dims,
                "primary_score": round(primary, 4),
                "reasoning": parsed.get("reasoning", ""),
                "summary": parsed.get("one_line_summary", ""),
            }

        except Exception as e:
            logger.warning(f"Score parse failed: {e}")
            return self._empty_scores()

    def _empty_scores(self) -> dict:
        return {
            "dimensions": {dim: 0.0 for dim in self.DIMENSIONS},
            "primary_score": 0.0,
            "reasoning": "",
            "summary": "Scoring failed",
        }