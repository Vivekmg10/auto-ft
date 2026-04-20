import json
from abc import ABC, abstractmethod
from typing import Any
from loguru import logger
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential


class BaseAgent(ABC):
    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: list[dict] | None = None,
        max_iterations: int = 10,
        temperature: float = 0.7,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.tool_registry: dict[str, callable] = {}
        self.max_iterations = max_iterations
        self.temperature = temperature

    def run(self, task: str, context: dict = {}) -> dict:
        """
        Main entry point. Builds prompt, runs the agent loop,
        returns structured output.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_prompt(task, context)}
        ]

        logger.debug(f"{self.__class__.__name__} starting task: {task[:60]}...")

        for iteration in range(self.max_iterations):
            response = self._call_llm(messages)
            message = response.choices[0].message

            # no tool call — agent is done
            if not message.tool_calls:
                logger.debug(f"{self.__class__.__name__} finished in {iteration + 1} iterations")
                return self._parse_output(message.content)

            # agent wants to call tools
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                logger.debug(f"Tool call: {tool_name}({tool_args})")

                result = self._dispatch(tool_name, tool_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result) if isinstance(result, dict) else str(result)
                })

        logger.warning(f"{self.__class__.__name__} hit max iterations ({self.max_iterations})")
        return self._parse_output(messages[-1].get("content", ""))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_llm(self, messages: list[dict]) -> Any:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"

        return completion(**kwargs)

    def _dispatch(self, tool_name: str, tool_args: dict) -> Any:
        if tool_name not in self.tool_registry:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return self.tool_registry[tool_name](**tool_args)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"error": str(e)}

    @abstractmethod
    def _build_prompt(self, task: str, context: dict) -> str:
        pass

    @abstractmethod
    def _parse_output(self, content: str) -> dict:
        pass