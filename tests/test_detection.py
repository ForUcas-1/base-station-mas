"""Tests for Detection Worker contracts and logic."""

import numpy as np
import pytest


class TestDetectionResult:
    """Test the DetectionResult Pydantic schema."""

    def test_valid_detection(self):
        from contracts.detection import DetectionResult

        result = DetectionResult(
            has_anomaly=True,
            anomaly_score=0.95,
            affected_kpis=["RSRP", "DL_BLER"],
            sample_index=42,
        )
        assert result.has_anomaly is True
        assert result.anomaly_score == 0.95
        assert len(result.affected_kpis) == 2

    def test_no_anomaly_detection(self):
        from contracts.detection import DetectionResult

        result = DetectionResult(
            has_anomaly=False,
            anomaly_score=0.03,
            affected_kpis=[],
            sample_index=0,
        )
        assert result.has_anomaly is False

    def test_score_clamped(self):
        """Pydantic should reject scores outside [0, 1]."""
        from contracts.detection import DetectionResult
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DetectionResult(
                has_anomaly=True,
                anomaly_score=1.5,
                affected_kpis=[],
                sample_index=0,
            )


class TestDiagnosisResult:
    """Test the DiagnosisResult Pydantic schema."""

    def test_skipped_diagnosis(self):
        from contracts.diagnosis import DiagnosisResult

        result = DiagnosisResult(
            skipped=True,
            skip_reason="No anomaly detected",
        )
        assert result.skipped is True
        assert result.root_cause is None

    def test_jamming_diagnosis(self):
        from contracts.diagnosis import (
            DiagnosisResult, MLCandidate, MLCandidates, GraphRAGContext,
        )

        ml = MLCandidates(candidates=[
            MLCandidate(label="Jamming", probability=0.52),
            MLCandidate(label="Co-Channel Interference (Mild)", probability=0.31),
            MLCandidate(label="High Network Congestion (Sudden Spike)", probability=0.10),
        ])

        ctx = GraphRAGContext(
            interference_source="Jammer",
            impact_paths=[],
            mobility_events=["UE moving Zone_A → Zone_B"],
            raw_context="Jammer → BS → Zone_B",
        )

        result = DiagnosisResult(
            skipped=False,
            root_cause="Jamming",
            confidence=0.94,
            ml_top3=ml,
            graphrag_context=ctx,
            reasoning="ML Top-1=Jamming(52%). GraphRAG confirms Jammer active.",
            topology_evidence=["Jammer → BS → Zone_B"],
        )

        assert result.root_cause == "Jamming"
        assert result.confidence == 0.94
        assert len(result.ml_top3.candidates) == 3


class TestOrchestratorTask:
    """Test the OrchestratorTask schema."""

    def test_task_creation(self):
        from contracts.task import OrchestratorTask

        task = OrchestratorTask(
            user_query="Zone B RSRP 为什么恶化？",
            intent="diagnose",
            zone="B",
        )
        assert task.user_query
        assert len(task.task_id) > 0  # UUID auto-generated
        assert task.zone == "B"

    def test_context_accumulation(self):
        from contracts.task import OrchestratorTask
        from contracts.detection import DetectionResult

        task = OrchestratorTask(user_query="test")
        det = DetectionResult(
            has_anomaly=True,
            anomaly_score=0.8,
            affected_kpis=["RSRP"],
            sample_index=10,
        )
        task.context.detection_result = det

        assert task.context.detection_result.has_anomaly is True
        assert task.context.diagnosis_result is None


class TestHeads:
    """Test model head factory."""

    def test_anomaly_detection_head(self):
        from src.models.heads import build_head
        import torch

        head = build_head("anomaly detection", d_model=32)
        x = torch.randn(4, 32)
        out = head(x)
        assert out.shape == (4, 2)

    def test_root_cause_head_11_classes(self):
        from src.models.heads import build_head
        import torch

        head = build_head("root-cause analysis", d_model=32)
        x = torch.randn(4, 32)
        out = head(x)
        assert out.shape == (4, 11)  # 11 classes including Jamming!

    def test_anomaly_duration_head(self):
        from src.models.heads import build_head
        import torch

        head = build_head("anomaly duration", d_model=32, seq_len=128)
        x = torch.randn(4, 32)
        out = head(x)
        assert out.shape == (4, 128)
        assert (out >= 0).all() and (out <= 1).all()  # Sigmoid output
