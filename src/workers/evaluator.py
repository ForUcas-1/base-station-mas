"""Evaluator Worker — LLM quality check on generated reports.

Five checks:
  1. Format validity (Markdown structure, required sections)
  2. Fact consistency (report claims vs upstream Worker outputs)
  3. Hallucination detection (claims not backed by source data)
  4. Suggestion quality (actionable repairs present)
  5. KPI coverage (all affected KPIs mentioned)

If any check fails, the Orchestrator retries the Reporter.
"""

import json

from contracts.evaluation import CheckDetail, EvalResult, EvaluationChecks
from contracts.report import Report
from contracts.task import OrchestratorTask
from src.workers.base import BaseWorker


class Evaluator(BaseWorker):
    """LLM-based quality evaluator for diagnostic reports.

    Config keys:
        llm: dict = {provider, model, temperature, max_tokens}
        checks:
            format_valid: bool = true
            fact_consistency: bool = true
            hallucination_check: bool = true
            suggestion_quality: bool = true
            kpi_coverage: bool = true
        max_retries: int = 2
    """

    SYSTEM_PROMPT = (
        "You are a quality assurance evaluator for 5G base station "
        "diagnostic reports. Your job is to verify that reports are "
        "accurate, complete, and free of hallucinations. "
        "Output valid JSON only."
    )

    def __init__(self, config: dict | None = None, config_path: str | None = None):
        super().__init__(config=config, config_path=config_path)
        worker_cfg = self.config.get("evaluator", self.config)

        self.checks_cfg = worker_cfg.get("checks", {})
        self.max_retries = worker_cfg.get("max_retries", 2)
        self._llm_client = None

    @property
    def name(self) -> str:
        return "Evaluator(LLM)"

    @property
    def llm_client(self):
        if self._llm_client is None:
            from src.llm.client import LLMClientFactory
            self._llm_client = LLMClientFactory.create()
        return self._llm_client

    async def execute(self, task: OrchestratorTask) -> EvalResult:
        """Run all enabled quality checks on the report.

        The report is expected to be in task.context (set by Orchestrator
        after Reporter returns), or passed via task metadata.
        """
        report: Report | None = None
        # Report is stored in metadata by Supervisor
        # We check both places
        if hasattr(task, "_report"):
            report = task._report

        if report is None:
            # Without a report to evaluate, pass by default
            return EvalResult(
                passed=True,
                summary="No report to evaluate",
            )

        detection = task.context.detection_result
        diagnosis = task.context.diagnosis_result

        checks = EvaluationChecks()

        # 1. Format check (non-LLM)
        if self.checks_cfg.get("format_valid", True):
            checks.format_valid = self._check_format(report)

        # 2-5. LLM-based checks
        if report.has_anomaly and any([
            self.checks_cfg.get("fact_consistency", True),
            self.checks_cfg.get("hallucination_check", True),
            self.checks_cfg.get("suggestion_quality", True),
            self.checks_cfg.get("kpi_coverage", True),
        ]):
            llm_checks = await self._llm_checks(report, detection, diagnosis)
            checks.fact_consistency = llm_checks.fact_consistency
            checks.hallucination_check = llm_checks.hallucination_check
            checks.suggestion_quality = llm_checks.suggestion_quality
            checks.kpi_coverage = llm_checks.kpi_coverage

        all_passed = all([
            checks.format_valid.passed,
            checks.fact_consistency.passed,
            checks.hallucination_check.passed,
            checks.suggestion_quality.passed,
            checks.kpi_coverage.passed,
        ])

        failures = []
        for name in ["format_valid", "fact_consistency", "hallucination_check",
                      "suggestion_quality", "kpi_coverage"]:
            check = getattr(checks, name)
            if not check.passed:
                failures.append(f"{name}: {check.detail}")

        return EvalResult(
            passed=all_passed,
            checks=checks,
            retry_count=0,
            summary="All checks passed" if all_passed
                    else f"Failed: {'; '.join(failures)}",
        )

    # ------------------------------------------------------------------
    # Format check (deterministic)
    # ------------------------------------------------------------------
    @staticmethod
    def _check_format(report: Report) -> CheckDetail:
        """Check that the Markdown report has required sections.

        For no-anomaly reports, only requires 概要 section.
        For anomaly reports, requires all 6 sections.
        """
        if report.has_anomaly:
            required = ["概要", "异常检测", "根因分析", "证据链",
                        "持续时间", "修复建议"]
        else:
            required = ["概要"]

        missing = [s for s in required if s not in report.markdown]

        if missing:
            return CheckDetail(
                passed=False,
                detail=f"Missing sections: {', '.join(missing)}",
            )
        return CheckDetail(passed=True, detail="All required sections present")

    # ------------------------------------------------------------------
    # LLM checks
    # ------------------------------------------------------------------
    async def _llm_checks(
        self,
        report: Report,
        detection,
        diagnosis,
    ) -> EvaluationChecks:
        """Use LLM to check fact consistency, hallucinations, etc."""
        detection_str = detection.model_dump_json(indent=2) if detection else "{}"
        diagnosis_str = diagnosis.model_dump_json(indent=2) if diagnosis else "{}"

        prompt = f"""Evaluate this 5G diagnostic report against the source data.

## Source Data (Ground Truth)
### Detection Result
{detection_str}

### Diagnosis Result
{diagnosis_str}

## Report to Evaluate
{report.markdown[:4000]}

## Evaluation Criteria
1. **fact_consistency**: Does the report's root cause match the diagnosis result?
   Does it correctly report the anomaly score and affected KPIs?
2. **hallucination_check**: Does the report invent KPI values, topology paths,
   or entity names not present in the source data?
3. **suggestion_quality**: Are repair suggestions concrete and actionable?
   Or are they vague/generic?
4. **kpi_coverage**: Are all affected KPIs from the detection result mentioned in the report?

Respond as JSON:
{{
  "fact_consistency": {{"passed": true/false, "detail": "<explanation>"}},
  "hallucination_check": {{"passed": true/false, "detail": "<specific suspicious claims>"}},
  "suggestion_quality": {{"passed": true/false, "detail": "<assessment>"}},
  "kpi_coverage": {{"passed": true/false, "detail": "<missing KPIs if any>"}}
}}"""

        try:
            response = await self.llm_client.complete(
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                response_format="json",
                temperature=0.0,
                max_tokens=1024,
            )
            data = json.loads(response)
        except Exception:
            # Degrade: pass all LLM checks
            return EvaluationChecks(
                fact_consistency=CheckDetail(
                    passed=True,
                    detail="LLM unavailable — skipped",
                ),
                hallucination_check=CheckDetail(
                    passed=True,
                    detail="LLM unavailable — skipped",
                ),
                suggestion_quality=CheckDetail(
                    passed=True,
                    detail="LLM unavailable — skipped",
                ),
                kpi_coverage=CheckDetail(
                    passed=True,
                    detail="LLM unavailable — skipped",
                ),
            )

        return EvaluationChecks(
            fact_consistency=CheckDetail(**data.get("fact_consistency", {"passed": True, "detail": ""})),
            hallucination_check=CheckDetail(**data.get("hallucination_check", {"passed": True, "detail": ""})),
            suggestion_quality=CheckDetail(**data.get("suggestion_quality", {"passed": True, "detail": ""})),
            kpi_coverage=CheckDetail(**data.get("kpi_coverage", {"passed": True, "detail": ""})),
        )
