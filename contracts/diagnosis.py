"""Diagnosis Worker output schemas."""

from pydantic import BaseModel, Field


class MLCandidate(BaseModel):
    """A single root cause candidate from ML classification."""

    label: str = Field(..., description="Root cause label, e.g. 'Jamming'")
    probability: float = Field(..., ge=0.0, le=1.0)


class MLCandidates(BaseModel):
    """Top-K root cause candidates from Step 1 (ML classification)."""

    candidates: list[MLCandidate] = Field(
        ..., min_length=1, max_length=3, description="Top-3 candidates"
    )
    encoder_used: str = Field(default="Autoformer")


class EntityState(BaseModel):
    """State of a knowledge graph entity at query time."""

    entity_type: str = Field(..., description="e.g. 'Jammer', 'BaseStation', 'Zone_B'")
    entity_id: str = Field(..., description="Node ID in the KG")
    attributes: dict = Field(default_factory=dict)


class TopologyPath(BaseModel):
    """A path through the knowledge graph showing anomaly propagation."""

    path: list[str] = Field(
        ..., description="Ordered node IDs, e.g. ['Jammer', 'BS', 'Zone_B']"
    )
    edges: list[str] = Field(
        default_factory=list,
        description="Edge types along the path, e.g. ['interference', 'covers']",
    )


class HistoricalCase(BaseModel):
    """A similar troubleshooting case retrieved by vector search."""

    ticket_id: int | None = None
    anomaly_type: str = ""
    description: str = ""
    similarity: float = Field(default=0.0, ge=0.0, le=1.0)


class GraphRAGContext(BaseModel):
    """Structured context from GraphRAG query (Step 2 output)."""

    interference_source: str | None = Field(
        None, description="Name of the interference source, e.g. 'Jammer'"
    )
    impact_paths: list[TopologyPath] = Field(
        default_factory=list,
        description="Entity relation paths showing anomaly propagation",
    )
    entity_states: list[EntityState] = Field(
        default_factory=list,
        description="Current state of relevant KG entities",
    )
    similar_cases: list[HistoricalCase] = Field(
        default_factory=list,
        description="Top-K similar historical troubleshooting cases",
    )
    mobility_events: list[str] = Field(
        default_factory=list,
        description="Mobile terminal mobility events, e.g. 'UE moving Zone A -> Zone B'",
    )
    raw_context: str = Field(
        default="",
        description="Serialized natural language summary for LLM prompt",
    )


class AnomalyDuration(BaseModel):
    """Estimated anomaly time window."""

    start_timestep: int = Field(..., ge=0, le=127)
    end_timestep: int = Field(..., ge=0, le=127)
    duration_seconds: float | None = None


_ANOMALY_TYPES = [
    "Antenna Failure",
    "Co-Channel Interference (Mild)",
    "Co-Channel Interference (Severe)",
    "Faulty RF Filters (Temporal)",
    "Doppler Shift (Severe)",
    "Faulty Handover Algorithm (Too Frequent)",
    "Buffer Overflow (Gradual Buildup)",
    "Resource Allocation Bugs",
    "High Network Congestion (Gradual Buildup)",
    "High Network Congestion (Sudden Spike)",
    "Jamming",
]


class DiagnosisResult(BaseModel):
    """Output from Diagnosis Worker (ML + GraphRAG + LLM).

    Contains the complete root cause analysis with evidence chain.
    """

    skipped: bool = Field(
        default=False,
        description="True if diagnosis was skipped (no anomaly detected)",
    )
    skip_reason: str = Field(
        default="",
        description="Reason for skipping, e.g. 'No anomaly detected'",
    )
    root_cause: str | None = Field(
        None, description="Final root cause label"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0
    )
    ml_top3: MLCandidates | None = Field(
        None, description="Top-3 ML candidates from Step 1"
    )
    duration: AnomalyDuration | None = None
    reasoning: str | None = Field(
        None, description="Chain-of-Thought reasoning from LLM (Step 3)"
    )
    topology_evidence: list[str] = Field(
        default_factory=list,
        description="Entity relation paths from GraphRAG, e.g. ['Jammer->BS->Zone_B']",
    )
    graphrag_context: GraphRAGContext | None = None
    encoder_used: str = Field(default="Autoformer")
