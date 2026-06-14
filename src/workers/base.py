"""Abstract base class for all Workers."""

from abc import ABC, abstractmethod

from contracts.task import OrchestratorTask
from src.utils.config import load_yaml_config


class BaseWorker(ABC):
    """Every Worker: load config → execute → return typed result.

    All workers run in-process (direct Python calls, no HTTP).
    In production, workers may be deployed as separate services,
    but the interface remains the same.
    """

    def __init__(self, config: dict | None = None, config_path: str | None = None):
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_yaml_config(config_path)
        else:
            raise ValueError("Either config or config_path must be provided")

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable worker identifier."""
        ...

    @abstractmethod
    async def execute(self, task: OrchestratorTask):
        """Execute the worker's core task.

        Args:
            task: The orchestrator task with accumulated upstream context.

        Returns:
            A Pydantic model (e.g., DetectionResult, DiagnosisResult, Report, EvalResult).
        """
        ...
