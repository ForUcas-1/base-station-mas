"""Worker contracts — centralized Pydantic schemas.

All Worker input/output types are defined here so that Workers,
Orchestrator, and API layer all share the same definitions.
"""

from contracts.detection import DetectionResult
from contracts.diagnosis import (
    AnomalyDuration,
    DiagnosisResult,
    EntityState,
    GraphRAGContext,
    HistoricalCase,
    MLCandidate,
    MLCandidates,
    TopologyPath,
)
from contracts.evaluation import CheckDetail, EvalResult, EvaluationChecks
from contracts.report import Report, ReportSections
from contracts.task import OrchestratorTask, RoutingRule, TaskContext

__all__ = [
    # Detection
    "DetectionResult",
    # Diagnosis
    "AnomalyDuration",
    "DiagnosisResult",
    "EntityState",
    "GraphRAGContext",
    "HistoricalCase",
    "MLCandidate",
    "MLCandidates",
    "TopologyPath",
    # Report
    "Report",
    "ReportSections",
    # Evaluation
    "CheckDetail",
    "EvalResult",
    "EvaluationChecks",
    # Task
    "OrchestratorTask",
    "RoutingRule",
    "TaskContext",
]
