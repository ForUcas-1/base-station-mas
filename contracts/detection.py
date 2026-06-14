"""Detection Worker output schema."""

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    """Output from Detection Worker (pure ML, TimesNet).

    Contains binary anomaly classification along with affected KPI details.
    """

    has_anomaly: bool = Field(
        ...,
        description="Whether an anomaly is detected in the KPI sequence",
    )
    anomaly_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (softmax probability of anomaly class)",
    )
    affected_kpis: list[str] = Field(
        default_factory=list,
        description="KPI names showing anomalous behavior, e.g. ['RSRP', 'DL_BLER']",
    )
    sample_index: int = Field(
        ...,
        ge=0,
        le=31999,
        description="Index in the TelecomTS dataset for traceability",
    )
    encoder_used: str = Field(
        default="TimesNet",
        description="Encoder model used for detection",
    )
    inference_time_ms: float = Field(
        default=0.0,
        description="Inference time in milliseconds",
    )
