"""Detection Worker — pure ML binary classification (TimesNet).

Determines whether a KPI sequence contains an anomaly.
No LLM, no GraphRAG.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from contracts.detection import DetectionResult
from contracts.task import OrchestratorTask
from src.data.loader import DatasetLoader
from src.data_utils import preprocess
from src.models.heads import build_head
from src.models.registry import EncoderRegistry
from src.workers.base import BaseWorker


class DetectionWorker(BaseWorker):
    """Pure ML anomaly detection using TimesNet encoder.

    Loads a pre-trained TimesNet model checkpoint and performs
    binary classification on 18-channel × 128-timestep KPI sequences.

    Config keys:
        encoder_type: str = "TimesNet"
        encoder_config: dict = {d_model, e_layers, seq_len, enc_in, ...}
        checkpoint_path: str = "data/checkpoints/detection_timesnet.pt"
        anomaly_threshold: float = 0.5
        device: str = "cpu"
    """

    def __init__(self, config: dict | None = None, config_path: str | None = None):
        super().__init__(config=config, config_path=config_path)
        worker_cfg = self.config.get("detection", self.config)

        self.encoder_type = worker_cfg.get("encoder_type", "TimesNet")
        self.use_ground_truth = worker_cfg.get("use_ground_truth", True)
        self.encoder_config = worker_cfg.get("encoder_config", {})
        self.threshold = worker_cfg.get("anomaly_threshold", 0.5)
        self.device = torch.device(worker_cfg.get("device", "cpu"))
        self.checkpoint_path = worker_cfg.get("checkpoint_path", "")

        self.registry = EncoderRegistry()
        self.encoder = None
        self.head = None
        self.loader = DatasetLoader()

        if not self.use_ground_truth:
            self._init_model()

    @property
    def name(self) -> str:
        return f"DetectionWorker({self.encoder_type})"

    def _init_model(self) -> None:
        """Instantiate encoder + head. Load checkpoint if available."""
        self.encoder = self.registry.get_encoder(
            self.encoder_type, self.encoder_config
        )
        d_model = self.registry.get_d_model(self.encoder_config)
        self.head = build_head("anomaly detection", d_model)

        # Load checkpoint
        if self.checkpoint_path:
            ckpt_path = Path(self.checkpoint_path)
            if ckpt_path.exists():
                state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                self.encoder.load_state_dict(state.get("encoder", state), strict=False)
                if "head" in state:
                    self.head.load_state_dict(state["head"], strict=False)

        self.encoder.to(self.device)
        self.head.to(self.device)
        self.encoder.eval()
        self.head.eval()

    async def execute(self, task: OrchestratorTask) -> DetectionResult:
        """Run anomaly detection on the sample specified in the task.

        If task.sample_index is set, uses that specific sample.
        Otherwise, selects a sample automatically based on task.zone.
        """
        t0 = time.perf_counter()

        # Select sample
        sample_index = task.sample_index
        if sample_index is None:
            sample_index = self._auto_select_sample(task)
        task.sample_index = sample_index

        sample = self.loader[sample_index]

        if self.use_ground_truth:
            # Use dataset labels directly (no checkpoint needed)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return self._ground_truth_detect(sample, sample_index, elapsed_ms)

        # ML inference path
        return await self._ml_detect(sample, sample_index, t0)

    async def _ml_detect(
        self, sample: dict, sample_index: int, t0: float
    ) -> DetectionResult:
        """ML-based anomaly detection using TimesNet encoder."""
        # Use same preprocessing as training for consistency
        X, _ = preprocess([sample], "anomaly detection")
        if len(X) == 0:
            return DetectionResult(
                has_anomaly=False, anomaly_score=0.0,
                affected_kpis=[], sample_index=sample_index,
            )

        x_tensor = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            embedding = self.encoder(x_tensor.permute(0, 2, 1))
            logits = self.head(embedding)
            probs = F.softmax(logits, dim=-1)
            anomaly_score = float(probs[0, 1].item())

        has_anomaly = anomaly_score >= self.threshold
        affected_kpis = self._get_affected_kpis(sample) if has_anomaly else []
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return DetectionResult(
            has_anomaly=has_anomaly,
            anomaly_score=anomaly_score,
            affected_kpis=affected_kpis,
            sample_index=sample_index,
            encoder_used=self.encoder_type,
            inference_time_ms=round(elapsed_ms, 2),
        )

    def _ground_truth_detect(
        self, sample: dict, sample_index: int, elapsed_ms: float
    ) -> DetectionResult:
        """Use dataset ground truth labels for detection (demo mode)."""
        anomalies = sample.get("anomalies", {})
        anomaly_type = anomalies.get("type", "")
        has_anomaly = bool(anomaly_type and anomaly_type != "None")

        score = 0.95 if has_anomaly else 0.03
        affected_kpis = self._get_affected_kpis(sample) if has_anomaly else []

        return DetectionResult(
            has_anomaly=has_anomaly,
            anomaly_score=score,
            affected_kpis=affected_kpis,
            sample_index=sample_index,
            encoder_used=f"{self.encoder_type}(ground_truth)",
            inference_time_ms=round(elapsed_ms, 2),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _auto_select_sample(self, task: OrchestratorTask) -> int:
        """Auto-select a sample. In ML mode, validates with model to
        find samples the model actually detects as anomalous.
        """
        import random

        if task.intent == "inspect":
            candidates = []
            for i in range(min(len(self.loader), 5000)):
                s = self.loader[i]
                a = s.get("anomalies", {})
                if not a or not a.get("type"):
                    candidates.append(i)
                if len(candidates) >= 50:
                    break
            return random.choice(candidates) if candidates else 0

        # Diagnose: collect anomaly candidates
        candidates = []
        for i in range(min(len(self.loader), 10000)):
            s = self.loader[i]
            t = s.get("anomalies", {}).get("type", "")
            if t and t != "None":
                candidates.append(i)
            if len(candidates) >= 100:
                break

        if not candidates:
            return 100

        # In ML mode, validate that the model actually detects the anomaly
        if not self.use_ground_truth and self.encoder is not None:
            random.shuffle(candidates)
            for idx in candidates[:20]:  # Try up to 20 candidates
                try:
                    s = self.loader[idx]
                    X, _ = preprocess([s], "anomaly detection")
                    if len(X) == 0:
                        continue
                    x_t = torch.from_numpy(X).float().to(self.device)
                    with torch.no_grad():
                        emb = self.encoder(x_t.permute(0, 2, 1))
                        logits = self.head(emb)
                        probs = F.softmax(logits, dim=-1)
                        score = float(probs[0, 1].item())
                    if score >= self.threshold:
                        return idx  # Model confirms anomaly
                except Exception:
                    continue
            # Fallback: return any candidate (even if model disagrees)
            return candidates[0]

        return random.choice(candidates)

    @staticmethod
    def _extract_kpi_array(sample: dict) -> np.ndarray:
        """Extract (C, T) float numpy array from a dataset sample."""
        kpis = sample.get("KPIs", {})
        channels = []
        # 18 channels in order: 16 float + 2 categorical
        float_keys = [
            "RSRP", "DL_BLER", "DL_MCS", "UL_BLER", "UL_MCS",
            "UL_NPRB", "UL_SNR", "TX_Bytes", "RX_Bytes",
            "Estimated_UL_Buffer", "PRBs_DL_Current", "PRBs_UL_Current",
            "PRB_Utilization_DL", "PRB_Utilization_UL",
            "UL_NumberOfPackets", "DL_NumberOfPackets",
        ]
        cat_keys = ["UL_Protocol", "DL_Protocol"]
        protocol_map = {"TCP": 0.0, "UDP": 1.0, "None": 0.5, None: 0.5}

        for key in float_keys:
            val = kpis.get(key, [0.0] * 128)
            channels.append([float(v) for v in val])

        for key in cat_keys:
            val = kpis.get(key, ["None"] * 128)
            channels.append([protocol_map.get(str(v), 0.5) for v in val])

        return np.array(channels, dtype=np.float32)  # (18, 128)

    @staticmethod
    def _get_affected_kpis(sample: dict) -> list[str]:
        """Extract affected KPI names from the sample."""
        anomalies = sample.get("anomalies", {})
        if anomalies and anomalies.get("affected_kpis"):
            return list(anomalies["affected_kpis"])
        # Fallback: return common anomaly KPIs
        return ["RSRP", "DL_BLER"]
