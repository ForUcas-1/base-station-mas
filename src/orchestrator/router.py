"""Intent Router — LLM-based query parsing to OrchestratorTask."""

import json

from contracts.task import OrchestratorTask, RoutingRule
from src.llm.client import LLMClientFactory


class IntentRouter:
    """LLM intent parser. Decomposes a natural language query into
    structured subtasks with conditional routing rules.

    Config keys:
        llm: dict = {provider, model, temperature, max_tokens}
    """

    DECOMPOSITION_PROMPT = """Parse the following 5G base station diagnostic query.
Output a JSON task plan.

## Query
{query}

## Output Format
{{
  "intent": "diagnose" | "inspect" | "question",
  "subtasks": ["detect", "diagnose", "report"],
  "routing": {{
    "diagnose": "if_has_anomaly" | "always" | "skip"
  }},
  "zone": "A" | "B" | "C" | null,
  "reasoning": "<one-line explanation>"
}}

## Rules
- "diagnose" intent: user asking about a specific anomaly/problem → subtasks=["detect","diagnose","report"], routing.diagnose="if_has_anomaly"
- "inspect" intent: user checking network status, no specific problem → subtasks=["detect","report"], routing.diagnose="if_has_anomaly"
- "question" intent: casual knowledge question, not about specific data → subtasks=["report"], routing.diagnose="skip"
- Extract zone from query if mentioned (A/B/C), otherwise null
"""

    def __init__(self, config: dict):
        self.config = config
        self.llm_client = LLMClientFactory.create()  # reads .env

    async def parse(self, user_query: str) -> OrchestratorTask:
        """Parse user query into a structured OrchestratorTask.

        Uses LLM for intent parsing with a rule-based fallback.
        """
        task = OrchestratorTask(user_query=user_query)

        try:
            response = await self.llm_client.complete(
                prompt=self.DECOMPOSITION_PROMPT.format(query=user_query),
                system="You parse 5G diagnostic queries into JSON task plans. "
                       "Output valid JSON only.",
                response_format="json",
                max_tokens=512,
                temperature=0.1,
            )
            data = json.loads(response)

            task.intent = data.get("intent", "diagnose")
            task.subtasks = data.get("subtasks", ["detect", "diagnose", "report"])
            task.routing = RoutingRule(
                diagnose=data.get("routing", {}).get("diagnose", "if_has_anomaly"),
            )
            task.zone = data.get("zone")

        except Exception:
            # Rule-based fallback
            task.intent = self._rule_based_intent(user_query)
            task.subtasks = self._rule_based_subtasks(task.intent)
            task.routing = RoutingRule(
                diagnose="skip" if task.intent == "question"
                else "if_has_anomaly"
            )
            task.zone = self._rule_based_zone(user_query)

        return task

    # ------------------------------------------------------------------
    # Rule-based fallback (no LLM required)
    # ------------------------------------------------------------------
    @staticmethod
    def _rule_based_intent(query: str) -> str:
        """Simple keyword-based intent classification."""
        q = query.lower()

        inspect_keywords = ["巡检", "检查", "状态", "正常", "扫描", "全量"]
        question_keywords = ["什么是", "为什么", "怎么", "如何", "解释", "原因"]

        if any(kw in q for kw in question_keywords) and not any(
            kw in q for kw in ["rsrp", "bler", "mcs", "kpi", "恶化", "异常", "告警"]
        ):
            return "question"

        if any(kw in q for kw in inspect_keywords):
            return "inspect"

        return "diagnose"

    @staticmethod
    def _rule_based_subtasks(intent: str) -> list[str]:
        """Determine subtasks from intent."""
        if intent == "question":
            return ["report"]
        if intent == "inspect":
            return ["detect", "report"]
        return ["detect", "diagnose", "report"]

    @staticmethod
    def _rule_based_zone(query: str) -> str | None:
        """Extract zone from query text."""
        q = query.upper()
        if "ZONE A" in q or "区域A" in q or "A区" in q:
            return "A"
        if "ZONE B" in q or "区域B" in q or "B区" in q:
            return "B"
        if "ZONE C" in q or "区域C" in q or "C区" in q:
            return "C"
        return None
