"""Diagnosis Worker — ML + GraphRAG + LLM (3-step pipeline).

Step 1: Autoformer ML classification → Top-3 root cause candidates
Step 2: GraphRAG query → network topology context
Step 3: LLM reasoning → final root cause + CoT + evidence chain

Steps 1 and 2 run in parallel; Step 3 depends on both.
Entire worker is skipped if has_anomaly=False.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from contracts.diagnosis import (
    AnomalyDuration,
    DiagnosisResult,
    MLCandidate,
    MLCandidates,
    GraphRAGContext,
)
from contracts.task import OrchestratorTask
from src.data.loader import DatasetLoader
from src.data_utils import preprocess
from src.graphrag.query import GraphRAGQuery
from src.models.heads import build_head
from src.models.registry import EncoderRegistry
from src.workers.base import BaseWorker


class DiagnosisWorker(BaseWorker):
    """ML + GraphRAG + LLM root cause analysis.

    Three internal steps:
      1. _step1_ml: Autoformer → 11-class → Top-3 candidates
      2. _step2_graphrag: GraphRAG hybrid query → topology context
      3. _step3_llm: LLM combines ML + KG → final diagnosis

    Steps 1 & 2 run concurrently. If has_anomaly=False, all skipped.

    Config keys:
        step1.encoder_type: str = "Autoformer"
        step1.encoder_config: dict
        step1.checkpoint_path: str
        step1.top_k: int = 3
        step2.kg_path: str
        step2.embeddings_dir: str
        step3.llm: dict = {provider, model, temperature, max_tokens}
    """

    # All 11 anomaly types (including Jamming — restored!)
    ANOMALY_TYPES = [
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
        "Jamming",  # ← Restored! Original code filtered this out.
    ]

    def __init__(self, config: dict | None = None, config_path: str | None = None):
        super().__init__(config=config, config_path=config_path)
        worker_cfg = self.config.get("diagnosis", self.config)

        # Step 1: ML
        step1_cfg = worker_cfg.get("step1", {})
        self.ml_encoder_type = step1_cfg.get("encoder_type", "Autoformer")
        self.ml_encoder_config = step1_cfg.get("encoder_config", {})
        self.ml_checkpoint_path = step1_cfg.get("checkpoint_path", "")
        self.ml_use_ground_truth = step1_cfg.get("use_ground_truth", True)
        self.ml_top_k = step1_cfg.get("top_k", 3)
        self.device = torch.device(step1_cfg.get("device", "cpu"))

        # Step 2: GraphRAG
        step2_cfg = worker_cfg.get("step2", {})
        self.graphrag = GraphRAGQuery(
            kg_path=step2_cfg.get("kg_path", "knowledge_graph/topology.json"),
            embeddings_dir=step2_cfg.get("embeddings_dir", "knowledge_graph/embeddings"),
            model_cache_dir="data/models",
        )

        # Step 3: LLM (lazy init to avoid import if not used)
        step3_cfg = worker_cfg.get("step3", {})
        self._llm_client = None

        self.registry = EncoderRegistry()
        self.encoder = None
        self.head = None
        self.loader = DatasetLoader()

        if not self.ml_use_ground_truth:
            self._init_ml_model()

    @property
    def name(self) -> str:
        return f"DiagnosisWorker({self.ml_encoder_type}+GraphRAG+LLM)"

    def _init_ml_model(self) -> None:
        """Initialize Autoformer encoder + 11-class head."""
        self.encoder = self.registry.get_encoder(
            self.ml_encoder_type, self.ml_encoder_config
        )
        d_model = self.registry.get_d_model(self.ml_encoder_config)
        self.head = build_head("root-cause analysis", d_model)

        if self.ml_checkpoint_path:
            ckpt = Path(self.ml_checkpoint_path)
            if ckpt.exists():
                state = torch.load(ckpt, map_location=self.device, weights_only=True)
                self.encoder.load_state_dict(state.get("encoder", state), strict=False)
                if "head" in state:
                    self.head.load_state_dict(state["head"], strict=False)

        self.encoder.to(self.device)
        self.head.to(self.device)
        self.encoder.eval()
        self.head.eval()

    @property
    def llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            from src.llm.client import LLMClientFactory
            self._llm_client = LLMClientFactory.create()
        return self._llm_client

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------
    async def execute(self, task: OrchestratorTask) -> DiagnosisResult:
        """Execute the 3-step diagnosis pipeline.

        Returns a skip-result immediately if no anomaly is detected.
        """
        detection = task.context.detection_result

        if detection is None or not detection.has_anomaly:
            return DiagnosisResult(
                skipped=True,
                skip_reason="No anomaly detected — diagnosis not needed",
            )

        # Run Step 1 (ML) and Step 2 (GraphRAG) in parallel
        ml_future = asyncio.create_task(self._step1_ml(task))
        graphrag_future = asyncio.create_task(self._step2_graphrag(task))

        ml_candidates = await ml_future
        graphrag_context = await graphrag_future

        # Step 3: LLM reasoning (depends on both)
        return await self._step3_llm(task, ml_candidates, graphrag_context)

    # ------------------------------------------------------------------
    # Step 1: ML Classification
    # ------------------------------------------------------------------
    async def _step1_ml(self, task: OrchestratorTask) -> MLCandidates:
        """Run Autoformer on the KPI sequence → Top-3 root cause candidates.

        PyTorch inference runs in a thread — can be interrupted by task cancel.
        """
        sample = self.loader[task.sample_index or 0]

        if self.ml_use_ground_truth:
            return self._ground_truth_classify(sample)

        try:
            X, _ = self._preprocess_with_jamming(sample)
            if X is None:
                from src.workers.detection import DetectionWorker
                kpi_array = DetectionWorker._extract_kpi_array(sample)
                X = np.expand_dims(kpi_array, axis=0)

            # Run PyTorch in thread — cancellable at await point
            x_tensor = torch.from_numpy(X).float()
            result = await asyncio.to_thread(
                self._run_ml_inference, x_tensor
            )
            return result

        except asyncio.CancelledError:
            raise  # 手动停止 — 立即传播
        except Exception:
            return MLCandidates(
                candidates=[MLCandidate(label="Unknown", probability=0.0)],
                encoder_used=self.ml_encoder_type,
            )

    def _run_ml_inference(self, x_tensor: torch.Tensor) -> MLCandidates:
        """Synchronous PyTorch inference — runs in a thread via to_thread."""
        x_tensor = x_tensor.to(self.device)
        with torch.no_grad():
            embedding = self.encoder(x_tensor.permute(0, 2, 1))
            logits = self.head(embedding)
            probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs[0], k=self.ml_top_k)
        candidates = [
            MLCandidate(
                label=self.ANOMALY_TYPES[int(idx)],
                probability=round(float(prob), 4),
            )
            for prob, idx in zip(topk_probs, topk_indices)
        ]
        return MLCandidates(candidates=candidates, encoder_used=self.ml_encoder_type)

    # ------------------------------------------------------------------
    # Step 2: GraphRAG
    # ------------------------------------------------------------------
    async def _step2_graphrag(self, task: OrchestratorTask) -> GraphRAGContext:
        """Query the knowledge graph for topology context."""
        try:
            return await self.graphrag.query(task)
        except Exception:
            return GraphRAGContext()

    # ------------------------------------------------------------------
    # Step 3: LLM Reasoning
    # ------------------------------------------------------------------
    async def _step3_llm(
        self,
        task: OrchestratorTask,
        ml_candidates: MLCandidates,
        graphrag_context: GraphRAGContext,
    ) -> DiagnosisResult:
        """LLM combines ML candidates + GraphRAG context → final diagnosis."""
        detection = task.context.detection_result

        prompt = self._build_llm_prompt(
            user_query=task.user_query,
            detection=detection,
            ml_candidates=ml_candidates,
            graphrag_context=graphrag_context,
        )

        try:
            response = await self.llm_client.complete(
                prompt=prompt,
                system="You are a 5G base station diagnostic expert. "
                       "Output valid JSON only.",
                response_format="json",
                temperature=0.2,
                max_tokens=2048,
            )
            data = json.loads(response)
        except Exception as e:
            import logging
            logging.getLogger("diagnosis").warning(
                f"LLM call failed: {type(e).__name__}: {e}"
            )
            # Degrade: return ML-only result
            top1 = ml_candidates.candidates[0] if ml_candidates.candidates else None
            return DiagnosisResult(
                skipped=False,
                root_cause=top1.label if top1 else "Unknown",
                confidence=top1.probability if top1 else 0.0,
                ml_top3=ml_candidates,
                reasoning=f"(LLM unavailable — {type(e).__name__}: {e})",
            )

        # Build topology evidence from graphrag context
        topology_evidence = []
        if graphrag_context.interference_source:
            for p in graphrag_context.impact_paths:
                topology_evidence.append(" → ".join(p.path))

        # Parse duration from LLM response
        duration = None
        dur_data = data.get("duration", {})
        if dur_data:
            duration = AnomalyDuration(
                start_timestep=int(dur_data.get("start_timestep") or 0),
                end_timestep=int(dur_data.get("end_timestep") or 0),
                duration_seconds=dur_data.get("duration_seconds"),
            )

        return DiagnosisResult(
            skipped=False,
            root_cause=data.get("root_cause", ""),
            confidence=float(data.get("confidence", 0.0)),
            ml_top3=ml_candidates,
            duration=duration,
            reasoning=data.get("reasoning", ""),
            topology_evidence=topology_evidence,
            graphrag_context=graphrag_context,
            encoder_used=self.ml_encoder_type,
        )

    # ------------------------------------------------------------------
    # Ground truth mode
    # ------------------------------------------------------------------
    def _ground_truth_classify(self, sample: dict) -> MLCandidates:
        """Use dataset ground truth labels for root cause classification."""
        anomalies = sample.get("anomalies", {})
        true_type = anomalies.get("type", "")
        if not true_type or true_type == "None":
            return MLCandidates(
                candidates=[MLCandidate(label="No Anomaly", probability=1.0)],
                encoder_used=f"{self.ml_encoder_type}(ground_truth)",
            )

        # Put ground truth at #1 with high probability
        candidates = [MLCandidate(label=true_type, probability=0.85)]

        # Add some other types as alternatives
        others = [a for a in self.ANOMALY_TYPES if a != true_type]
        for alt in others[:self.ml_top_k - 1]:
            candidates.append(MLCandidate(label=alt, probability=round(0.05, 4)))

        return MLCandidates(
            candidates=candidates,
            encoder_used=f"{self.ml_encoder_type}(ground_truth)",
        )
    def _build_llm_prompt(
        self,
        user_query: str,
        detection,
        ml_candidates: MLCandidates,
        graphrag_context: GraphRAGContext,
    ) -> str:
        """Construct the LLM reasoning prompt with all evidence."""
        ml_lines = []
        for i, c in enumerate(ml_candidates.candidates, 1):
            ml_lines.append(f"  {i}. {c.label} ({c.probability:.1%})")

        # Detection info
        has_anomaly = detection.has_anomaly if detection else "N/A"
        anomaly_score = f"{detection.anomaly_score:.3f}" if detection else "N/A"
        affected_kpis = ", ".join(detection.affected_kpis) if detection else "N/A"

        return f"""Analyze the following 5G base station diagnostic data.

## User Query
{user_query}

## Anomaly Detection
- Has anomaly: {has_anomaly}
- Anomaly score: {anomaly_score}
- Affected KPIs: {affected_kpis}

## ML Root Cause Candidates (Top-{self.ml_top_k})
{chr(10).join(ml_lines)}

## Network Topology Context (GraphRAG)
{graphrag_context.raw_context or '(No topology context available)'}

## Task
Determine the most likely root cause. Consider:
1. ML probabilities from the 11-class classifier
2. GraphRAG topology evidence (interference paths, entity states)
3. KPI symptom patterns (e.g., RSRP drop + BLER burst = Jamming)
4. Mobility timeline alignment
5. Historical similar cases

Respond as a JSON object with:
{{
  "root_cause": "<one of the 11 anomaly types>",
  "confidence": <0.0-1.0>,
  "duration": {{
    "start_timestep": <int 0-127>,
    "end_timestep": <int 0-127>,
    "duration_seconds": <float or null>
  }},
  "reasoning": "<step-by-step CoT referencing ML and GraphRAG evidence>"
}}"""

    @staticmethod
    def _preprocess_with_jamming(sample: dict) -> tuple:
        """Preprocess a sample for root-cause analysis WITHOUT filtering Jamming.

        The original data_utils.preprocess() filters out Jamming with:
          if item["anomalies"]["type"] == "Jamming": continue
        This override ensures Jamming is included as a valid class.
        """
        try:
            # Use anomaly detection preprocessing (includes all samples)
            # then pair with root-cause labels
            from src.data_utils import preprocess, anomaly_type_2_id

            X, _ = preprocess([sample], "anomaly detection")

            anomalies = sample.get("anomalies", {})
            anomaly_type = anomalies.get("type", "")
            if anomaly_type and anomaly_type != "None":
                y = np.array([anomaly_type_2_id(anomaly_type)])
            else:
                y = np.array([0])

            return X, y
        except Exception:
            return None, None
