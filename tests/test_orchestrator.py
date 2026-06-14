"""Tests for orchestrator routing logic (no LLM required)."""

import pytest


class TestIntentRouter:
    """Test the rule-based fallback of IntentRouter (no LLM)."""

    def test_diagnose_intent(self):
        from src.orchestrator.router import IntentRouter

        intent = IntentRouter._rule_based_intent("Zone B RSRP 为什么恶化")
        assert intent == "diagnose"

        subtasks = IntentRouter._rule_based_subtasks(intent)
        assert subtasks == ["detect", "diagnose", "report"]

    def test_inspect_intent(self):
        from src.orchestrator.router import IntentRouter

        intent = IntentRouter._rule_based_intent("网络状态巡检")
        assert intent == "inspect"

        subtasks = IntentRouter._rule_based_subtasks(intent)
        assert subtasks == ["detect", "report"]  # No diagnosis

    def test_question_intent(self):
        from src.orchestrator.router import IntentRouter

        intent = IntentRouter._rule_based_intent("RSRP 下降通常是什么原因")
        assert intent == "question"

        subtasks = IntentRouter._rule_based_subtasks(intent)
        assert subtasks == ["report"]  # Only report

    def test_zone_extraction(self):
        from src.orchestrator.router import IntentRouter

        assert IntentRouter._rule_based_zone("Zone A 有问题") == "A"
        assert IntentRouter._rule_based_zone("B区异常") == "B"
        assert IntentRouter._rule_based_zone("整个网络正常") is None


class TestSupervisorResponse:
    """Test response assembly."""

    def test_build_response(self):
        from contracts.detection import DetectionResult
        from contracts.diagnosis import DiagnosisResult
        from contracts.evaluation import EvalResult
        from contracts.report import Report
        from contracts.task import OrchestratorTask
        from src.orchestrator.supervisor import Supervisor

        task = OrchestratorTask(
            user_query="test",
            intent="diagnose",
            zone="B",
        )
        det = DetectionResult(
            has_anomaly=True,
            anomaly_score=0.95,
            affected_kpis=["RSRP"],
            sample_index=42,
        )
        diag = DiagnosisResult(
            skipped=False,
            root_cause="Jamming",
            confidence=0.94,
        )
        report = Report(
            markdown="# Test Report\n\nAll clear.",
            has_anomaly=True,
        )
        ev = EvalResult(passed=True)

        response = Supervisor._build_response(
            task=task,
            detection=det,
            diagnosis=diag,
            report=report,
            evaluation=ev,
            elapsed_ms=1234.5,
        )

        assert response["has_anomaly"] is True
        assert response["root_cause"] == "Jamming"
        assert response["eval_passed"] is True
        assert "Test Report" in response["report_markdown"]
        assert response["latency_ms"] == 1234.5
