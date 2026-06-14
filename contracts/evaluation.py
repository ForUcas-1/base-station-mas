"""Evaluator Worker output schema."""

from pydantic import BaseModel, Field


class CheckDetail(BaseModel):
    """Result of a single evaluation check."""

    passed: bool = Field(..., description="Whether this check passed")
    detail: str = Field(default="", description="Human-readable explanation")


class EvaluationChecks(BaseModel):
    """All evaluation check results."""

    format_valid: CheckDetail = Field(
        default_factory=lambda: CheckDetail(passed=True, detail="")
    )
    fact_consistency: CheckDetail = Field(
        default_factory=lambda: CheckDetail(passed=True, detail="")
    )
    hallucination_check: CheckDetail = Field(
        default_factory=lambda: CheckDetail(passed=True, detail="")
    )
    suggestion_quality: CheckDetail = Field(
        default_factory=lambda: CheckDetail(passed=True, detail="")
    )
    kpi_coverage: CheckDetail = Field(
        default_factory=lambda: CheckDetail(passed=True, detail="")
    )


class EvalResult(BaseModel):
    """Output from Evaluator Worker (LLM quality check).

    If passed=False, the Orchestrator will retry the Reporter.
    """

    passed: bool = Field(
        ...,
        description="True if all required checks pass",
    )
    checks: EvaluationChecks = Field(default_factory=EvaluationChecks)
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retries so far",
    )
    summary: str = Field(
        default="",
        description="One-line summary of evaluation outcome",
    )
