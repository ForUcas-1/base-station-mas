"""Reporter Worker — LLM-based Markdown diagnostic report generation.

Generates a comprehensive diagnostic report in the style of
TelecomTS QnA reasoning chains, with optional few-shot examples.
"""

import random

from contracts.report import Report, ReportSections
from contracts.task import OrchestratorTask
from src.data.loader import DatasetLoader
from src.workers.base import BaseWorker


class ReporterWorker(BaseWorker):
    """LLM report generation referencing TelecomTS QnA reasoning style.

    For anomaly cases: generates 6-section Markdown report with evidence.
    For normal cases: generates a simple "all clear" status report.

    Config keys:
        llm: dict = {provider, model, temperature, max_tokens}
        report_style:
            template: markdown
            sections: [summary, anomaly_detection, ...]
        few_shot:
            enabled: bool
            num_examples: int
    """

    SYSTEM_PROMPT = (
        "You are a 5G base station diagnostic report writer. "
        "Write in clear Chinese technical prose suitable for field "
        "maintenance engineers. Reference specific KPIs, timesteps, "
        "and topology evidence. Include actionable repair suggestions. "
        "Output valid JSON only."
    )

    def __init__(self, config: dict | None = None, config_path: str | None = None):
        super().__init__(config=config, config_path=config_path)
        worker_cfg = self.config.get("reporter", self.config)

        self.report_style = worker_cfg.get("report_style", {})
        self.few_shot_enabled = worker_cfg.get("few_shot", {}).get("enabled", False)
        self.few_shot_num = worker_cfg.get("few_shot", {}).get("num_examples", 3)

        self.loader = DatasetLoader()
        self._llm_client = None

    @property
    def name(self) -> str:
        return "ReporterWorker(LLM)"

    @property
    def llm_client(self):
        if self._llm_client is None:
            from src.llm.client import LLMClientFactory
            self._llm_client = LLMClientFactory.create()
        return self._llm_client

    async def execute(self, task: OrchestratorTask) -> Report:
        """Generate a diagnostic report from all upstream results."""
        detection = task.context.detection_result
        diagnosis = task.context.diagnosis_result

        if detection is None or not detection.has_anomaly:
            return self._normal_report(task)

        # Load few-shot examples
        few_shot = ""
        if self.few_shot_enabled:
            few_shot = self._load_few_shot_examples()

        # Build prompt
        prompt = self._build_prompt(task, detection, diagnosis, few_shot)

        try:
            response = await self.llm_client.complete(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                response_format="json",
                temperature=0.4,
                max_tokens=3072,
            )
            import json
            data = json.loads(response)
        except Exception as e:
            import logging
            logging.getLogger("reporter").warning(
                f"LLM call failed: {type(e).__name__}: {e}"
            )
            # Degrade: build report from structured data without LLM
            data = self._build_fallback_report(task, detection, diagnosis)

        sections = ReportSections(
            summary=data.get("summary", ""),
            anomaly_detection=data.get("anomaly_detection", ""),
            root_cause_analysis=data.get("root_cause_analysis", ""),
            evidence_chain=data.get("evidence_chain", ""),
            duration_impact=data.get("duration_impact", ""),
            recommendations=data.get("recommendations", ""),
        )

        markdown = self._assemble_markdown(sections, detection, diagnosis)

        return Report(
            markdown=markdown,
            sections=sections,
            has_anomaly=True,
            repair_suggestions=data.get("repair_suggestions", []),
        )

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        task: OrchestratorTask,
        detection,
        diagnosis,
        few_shot: str,
    ) -> str:
        """Build the full reporter prompt with all evidence."""
        detection_json = detection.model_dump_json(indent=2) if detection else "{}"
        diagnosis_json = diagnosis.model_dump_json(indent=2) if diagnosis else "{}"

        return f"""Write a 5G base station diagnostic report in Chinese.

## User Query
{task.user_query}

## Detection Result
{detection_json}

## Diagnosis Result
{diagnosis_json}

{few_shot}

## Report Sections Required
Generate a JSON object with these string fields (all in Chinese):
{{
  "summary": "<1-2 sentence executive summary>",
  "anomaly_detection": "<description of what anomaly was found, which KPIs, severity>",
  "root_cause_analysis": "<which root cause, confidence, how determined, ML+KG evidence>",
  "evidence_chain": "<step-by-step reasoning chain from detection through topology to conclusion>",
  "duration_impact": "<how long the anomaly lasted, which timesteps, impact on network>",
  "recommendations": "<actionable repair suggestions for field engineer>",
  "repair_suggestions": ["<suggestion 1>", "<suggestion 2>"]
}}

## Style Guidelines
- Use professional but accessible Chinese
- Reference specific KPI values and timesteps
- Cite topology evidence (e.g., "Jammer→BS→Zone_B")
- Include concrete repair steps a field engineer can execute
- Use 👍/⚠️/🔴 emoji sparingly for visual clarity"""

    # ------------------------------------------------------------------
    # Few-shot
    # ------------------------------------------------------------------
    def _load_few_shot_examples(self) -> str:
        """Load few-shot examples from TelecomTS QnA anomalies."""
        examples = []
        max_samples = min(len(self.loader), 500)
        for i in random.sample(range(max_samples), min(self.few_shot_num * 5, max_samples)):
            sample = self.loader[i]
            qna = sample.get("QnA", [])
            anomalies_qna = [
                q for q in qna
                if isinstance(q, dict) and q.get("category") == "anomalies"
            ]
            if anomalies_qna:
                q = random.choice(anomalies_qna)
                examples.append({
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "reasoning": q.get("reasoning", ""),
                })
            if len(examples) >= self.few_shot_num:
                break

        if not examples:
            return ""

        lines = ["## Few-Shot Examples (TelecomTS QnA Style)"]
        for i, ex in enumerate(examples, 1):
            lines.append(f"### Example {i}")
            lines.append(f"Q: {ex['question']}")
            lines.append(f"A: {ex['answer']}")
            if ex.get("reasoning"):
                lines.append(f"Reasoning: {ex['reasoning'][:300]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Markdown assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _assemble_markdown(
        sections: ReportSections,
        detection,
        diagnosis,
    ) -> str:
        """Assemble sections into a complete Markdown report."""
        anomaly_score = detection.anomaly_score if detection else 0.0
        root_cause = diagnosis.root_cause if diagnosis else "Unknown"
        confidence = diagnosis.confidence if diagnosis else 0.0
        affected = ", ".join(detection.affected_kpis) if detection else "N/A"

        return f"""# 📊 5G 基站诊断报告

## 📋 概要
{sections.summary}

---

## ⚠️ 异常检测
{sections.anomaly_detection}

| 指标 | 值 |
|------|----|
| 异常分数 | {anomaly_score:.1%} |
| 受影响 KPI | {affected} |
| 根因 | **{root_cause}** |
| 置信度 | {confidence:.1%} |

## 🛠️ 根因分析
{sections.root_cause_analysis}

## 🔗 证据链
{sections.evidence_chain}

## ⏱️ 持续时间与影响
{sections.duration_impact}

## 💡 修复建议
{sections.recommendations}

---
*报告由 BaseStation-MAS Multi-Agent 系统自动生成*
"""

    # ------------------------------------------------------------------
    # Normal / fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _normal_report(task: OrchestratorTask) -> Report:
        """Simple report for normal (no-anomaly) case."""
        markdown = f"""# ✅ 5G 基站巡检报告

## 📋 概要
全网巡检未发现异常，所有 KPI 指标在正常范围内。

## 📊 状态
| 指标 | 状态 |
|------|------|
| 异常检测 | ✅ 正常 |
| 异常分数 | < 0.05 |

---
*报告由 BaseStation-MAS Multi-Agent 系统自动生成*
"""
        return Report(
            markdown=markdown,
            has_anomaly=False,
            sections=ReportSections(
                summary="全网正常，无异常。",
                anomaly_detection="未检测到异常。",
            ),
        )

    @staticmethod
    def _build_fallback_report(task, detection, diagnosis) -> dict:
        """Build report dict from structured data without LLM (degraded mode)."""
        root_cause = diagnosis.root_cause if diagnosis else "Unknown"
        reasoning = diagnosis.reasoning if diagnosis else ""
        evidence = diagnosis.topology_evidence if diagnosis else []

        return {
            "summary": f"检测到异常，根因判定为 {root_cause}。",
            "anomaly_detection": (
                f"异常分数 {detection.anomaly_score:.1%}，"
                f"受影响 KPI：{', '.join(detection.affected_kpis)}。"
            ),
            "root_cause_analysis": f"ML+GraphRAG 综合判定根因为 {root_cause}。",
            "evidence_chain": reasoning if reasoning else "（LLM 不可用，无推理链）",
            "duration_impact": "（无 LLM 可用，缺少时长估计）",
            "recommendations": f"请联系 NOC 排查 {root_cause} 相关问题。"
                + (" 拓扑证据：" + "; ".join(evidence) if evidence else ""),
            "repair_suggestions": [
                f"检查 {root_cause} 相关告警",
                "查看受影响的 Zone 历史 KPI 趋势",
            ],
        }
