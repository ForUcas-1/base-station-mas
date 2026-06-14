"""Reporter Worker output schema."""

from pydantic import BaseModel, Field


class ReportSections(BaseModel):
    """Individual sections of the diagnostic report."""

    summary: str = Field(default="", description="Executive summary")
    anomaly_detection: str = Field(default="", description="Anomaly detection findings")
    root_cause_analysis: str = Field(
        default="", description="Root cause determination"
    )
    evidence_chain: str = Field(default="", description="Evidence and reasoning chain")
    duration_impact: str = Field(default="", description="Duration and impact assessment")
    recommendations: str = Field(
        default="", description="Repair suggestions and next steps"
    )


class Report(BaseModel):
    """Output from Reporter Worker (LLM-generated Markdown report)."""

    markdown: str = Field(
        ...,
        description="Full Markdown diagnostic report",
    )
    sections: ReportSections = Field(default_factory=ReportSections)
    has_anomaly: bool = Field(default=False)
    repair_suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable repair suggestions",
    )
    source_qna_references: list[str] = Field(
        default_factory=list,
        description="TelecomTS QnA examples referenced for reasoning style",
    )
