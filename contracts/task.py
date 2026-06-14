"""Orchestrator Task schema."""

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from contracts.detection import DetectionResult
from contracts.diagnosis import DiagnosisResult


class RoutingRule(BaseModel):
    """Conditional routing rule for a subtask."""

    diagnose: str = Field(
        default="if_has_anomaly",
        description="When to run Diagnosis: 'if_has_anomaly' | 'always' | 'skip'",
    )


class TaskContext(BaseModel):
    """Accumulated results from upstream workers.

    Populated progressively as the pipeline executes.
    """

    detection_result: DetectionResult | None = None
    diagnosis_result: DiagnosisResult | None = None


class OrchestratorTask(BaseModel):
    """The task object passed through the entire pipeline.

    Created by IntentRouter, enriched by each Worker.
    """

    task_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique task identifier (UUID)",
    )
    user_query: str = Field(
        ...,
        description="Original natural language query from the user",
    )
    intent: str = Field(
        default="diagnose",
        description="Parsed intent: 'diagnose' | 'inspect' | 'question'",
    )
    subtasks: list[str] = Field(
        default_factory=lambda: ["detect", "diagnose", "report"],
        description="Ordered list of subtasks to execute",
    )
    routing: RoutingRule = Field(default_factory=RoutingRule)

    # Task scoping
    zone: str | None = Field(
        None, description="Target zone: 'A' | 'B' | 'C' | None (auto)"
    )
    sample_index: int | None = Field(
        None, ge=0, le=31999, description="Specific dataset sample, or None for auto-select"
    )

    # Accumulated context
    context: TaskContext = Field(default_factory=TaskContext)

    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 creation timestamp",
    )
