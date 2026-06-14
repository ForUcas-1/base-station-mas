"""Supervisor — DAG execution engine for the Multi-Agent pipeline."""

import asyncio
import time
from typing import Any

from contracts.detection import DetectionResult
from contracts.diagnosis import DiagnosisResult
from contracts.evaluation import EvalResult
from contracts.report import Report
from contracts.task import OrchestratorTask
from src.llm.client import LLMClientFactory
from src.orchestrator.router import IntentRouter
from src.utils.config import load_yaml_config
from src.workers.detection import DetectionWorker
from src.workers.diagnosis import DiagnosisWorker
from src.workers.reporter import ReporterWorker
from src.workers.evaluator import Evaluator


class Supervisor:
    """Orchestrates the full Multi-Agent diagnostic pipeline.

    All Workers run in-process for MVP simplicity.
    Each Worker is initialized once at startup.

    Usage:
        supervisor = Supervisor("configs/default.yaml")
        result = await supervisor.handle_query("Zone B RSRP为什么骤降?")
        print(result["report_markdown"])
    """

    def __init__(self, config_path: str):
        self.config = load_yaml_config(config_path)
        self.max_eval_retries = self.config.get("orchestrator", {}).get(
            "max_eval_retries", 2
        )

        # LLM for intent routing (reads LLM_PROVIDER + LLM_MODEL from .env)
        self.router = IntentRouter({})

        # Workers
        self.detection = DetectionWorker(config_path="configs/detection.yaml")
        self.diagnosis = DiagnosisWorker(config_path="configs/diagnosis.yaml")
        self.reporter = ReporterWorker(config_path="configs/reporter.yaml")
        self.evaluator = Evaluator(config_path="configs/evaluator.yaml")

        # Metrics
        self._worker_status: dict[str, str] = {}

        # Warm up models — no lazy loading during pipeline execution
        print("  Warming up models...")
        _ = self.diagnosis.graphrag.embedder.model  # SentenceTransformer
        print("  All models ready ✓")

    async def stop_all(self):
        """[手动停止] 关闭所有 Worker 的 LLM HTTP 连接，强制中断正在跑的请求。"""
        print("\n⏹ 手动停止 — 强制中断所有 Agent...")
        for name, worker in [
            ("diagnosis", self.diagnosis),
            ("reporter", self.reporter),
            ("evaluator", self.evaluator),
        ]:
            client = getattr(worker, '_llm_client', None)
            if client is not None:
                try:
                    await client.close()
                    print(f"  ✓ {name} LLM 连接已断开")
                except Exception as e:
                    print(f"  ⚠ {name}: {e}")
        print("  ⏹ 所有 Agent 已中断")

    async def handle_query(self, user_query: str) -> dict[str, Any]:
        """Full pipeline from natural language query."""
        t_start = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"  📨 收到查询: {user_query}")
        print(f"{'='*60}")

        # Phase 1: Intent parsing
        print(f"\n{'─'*40}")
        print(f"  🧠 [1/5] Orchestrator — 意图解析")
        task = await self.router.parse(user_query)
        print(f"    意图: {task.intent} | Zone: {task.zone or 'auto'} | 子任务: {task.subtasks}")
        return await self.handle_task(task, t_start)

    async def handle_task(
        self, task: OrchestratorTask, t_start: float | None = None,
        on_event = None,
    ) -> dict[str, Any]:
        """Run pipeline from a pre-built OrchestratorTask (skip intent parsing).

        Args:
            task: Pre-built task with sample_index set.
            t_start: Optional start timestamp.
            on_event: Optional callback(type, data) for real-time SSE push.
        """
        if t_start is None:
            t_start = time.perf_counter()

        def push(evt_type, data):
            if on_event:
                on_event(evt_type, data)

        self._worker_status["orchestrator"] = f"prebuilt intent={task.intent}"
        push("agent_status", {"agent": "Orchestrator", "status": "done"})

        # Phase 2: Anomaly Detection (skip if already done)
        det_result = task.context.detection_result
        if det_result is None:
            push("agent_status", {"agent": "Detection", "status": "running"})
            await asyncio.sleep(0.1)
            print(f"\n{'─'*40}")
            print(f"  📡 [2/5] Detection Worker — 异常检测")
            det_result = await self.detection.execute(task)
            task.context.detection_result = det_result
            # ── 实时推送 Detection 结果 ──
            push("detection_done", {
                "has_anomaly": det_result.has_anomaly,
                "anomaly_score": det_result.anomaly_score,
                "affected_kpis": det_result.affected_kpis,
                "sample_index": det_result.sample_index,
                "inference_time_ms": det_result.inference_time_ms,
            })
            await asyncio.sleep(0)
            push("agent_status", {"agent": "Detection", "status": "done"})
            print(f"    异常: {det_result.has_anomaly} | 分数: {det_result.anomaly_score:.3f}")
            print(f"    受影响 KPI: {det_result.affected_kpis}")
            self._worker_status["detection"] = (
                f"has_anomaly={det_result.has_anomaly} "
                f"score={det_result.anomaly_score:.3f}"
            )
        else:
            push("agent_status", {"agent": "Detection", "status": "done"})
            print(f"\n{'─'*40}")
            print(f"  📡 [2/5] Detection Worker — ⏭ 已有结果 (跳过)")
            self._worker_status["detection"] = "already done"

        # Phase 3: Diagnosis (conditional)
        should_diagnose = (
            "diagnose" in task.subtasks
            and task.routing.diagnose != "skip"
            and det_result.has_anomaly
        )

        diag_result = None
        if should_diagnose:
            push("agent_status", {"agent": "Diagnosis", "status": "running"})
            await asyncio.sleep(0.1)
            print(f"\n{'─'*40}")
            print(f"  🛠️ [3/5] Diagnosis Worker — 根因分析")
            diag_result = await self.diagnosis.execute(task)
            task.context.diagnosis_result = diag_result
            if diag_result.skipped:
                print(f"    ⏭ 跳过: {diag_result.skip_reason}")
            else:
                print(f"    根因: {diag_result.root_cause} | 置信度: {diag_result.confidence:.2f}")
                if diag_result.ml_top3:
                    top3_str = ", ".join(
                        f"{c.label}({c.probability:.1%})"
                        for c in diag_result.ml_top3.candidates
                    )
                    print(f"    ML Top-3: {top3_str}")
                if diag_result.topology_evidence:
                    print(f"    拓扑证据: {diag_result.topology_evidence}")
                print(f"    推理链: {(diag_result.reasoning or '')[:200]}...")
            self._worker_status["diagnosis"] = (
                f"root_cause={diag_result.root_cause} "
                f"confidence={diag_result.confidence:.2f}"
            )
        else:
            print(f"\n{'─'*40}")
            print(f"  🛠️ [3/5] Diagnosis Worker — ⏭ 跳过 (无异常或路由跳过)")
            self._worker_status["diagnosis"] = "skipped"

        push("agent_status", {"agent": "Diagnosis", "status": "done"})

        # Phase 4: Reporter — skip if no anomaly (正常时无需报告)
        report = None
        if det_result and det_result.has_anomaly:
            push("agent_status", {"agent": "Reporter", "status": "running"})
            await asyncio.sleep(0.1)
            print(f"\n{'─'*40}")
            print(f"  💬 [4/5] Reporter Worker — 报告生成")
            report = await self.reporter.execute(task)
            task._report = report
            push("agent_status", {"agent": "Reporter", "status": "done"})
            print(f"    报告长度: {len(report.markdown)} 字符")
            self._worker_status["reporter"] = f"markdown_len={len(report.markdown)}"

            # Phase 5: Evaluator
            push("agent_status", {"agent": "Evaluator", "status": "running"})
            await asyncio.sleep(0.1)
            print(f"\n{'─'*40}")
            print(f"  ✅ [5/5] Evaluator — 质量检查")
            eval_result = None
            for retry in range(self.max_eval_retries + 1):
                eval_result = await self.evaluator.execute(task)
                eval_result.retry_count = retry
                if eval_result.passed:
                    print(f"    ✓ 通过 (retry={retry})")
                    self._worker_status["evaluator"] = f"passed (retry={retry})"
                    break
                else:
                    print(f"    ✗ 不通过 (retry={retry}): {eval_result.summary}")
                    self._worker_status["evaluator"] = (
                        f"failed retry={retry}: {eval_result.summary}"
                    )
                    if retry < self.max_eval_retries:
                        report = await self.reporter.execute(task)
                        task._report = report
            push("agent_status", {"agent": "Evaluator", "status": "done"})
        else:
            eval_result = None
            print(f"\n{'─'*40}")
            print(f"  💬 [4/5] Reporter — ⏭ 跳过 (无异常无需报告)")
            print(f"  ✅ [5/5] Evaluator — ⏭ 跳过")
            self._worker_status["reporter"] = "skipped (normal)"
            self._worker_status["evaluator"] = "skipped (normal)"

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        print(f"\n{'='*60}")
        print(f"  🏁 流水线完成 | 耗时: {elapsed_ms:.0f}ms")
        print(f"{'='*60}\n")

        # Assemble response
        return self._build_response(
            task=task,
            detection=det_result,
            diagnosis=diag_result,
            report=report,
            evaluation=eval_result,
            elapsed_ms=elapsed_ms,
        )

    def worker_status(self) -> dict[str, str]:
        """Return per-worker status for health check."""
        return {
            "status": "healthy",
            "workers": self._worker_status,
        }

    # ------------------------------------------------------------------
    # Response assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _build_response(
        task: OrchestratorTask,
        detection: DetectionResult,
        diagnosis: DiagnosisResult | None,
        report: Report,
        evaluation: EvalResult | None,
        elapsed_ms: float,
    ) -> dict[str, Any]:
        """Assemble the final API response from all pipeline outputs."""
        response = {
            "task_id": task.task_id,
            "intent": task.intent,
            "zone": task.zone,
            "sample_index": task.sample_index,
            "has_anomaly": detection.has_anomaly,
            "anomaly_score": detection.anomaly_score,
            "affected_kpis": detection.affected_kpis,
            "report_markdown": report.markdown if report else "",
        }

        if diagnosis and not diagnosis.skipped:
            response.update({
                "root_cause": diagnosis.root_cause,
                "confidence": diagnosis.confidence,
                "reasoning": diagnosis.reasoning,
                "topology_evidence": diagnosis.topology_evidence,
                "ml_top3": [
                    {"label": c.label, "probability": c.probability}
                    for c in (diagnosis.ml_top3.candidates
                              if diagnosis.ml_top3 else [])
                ],
                "anomaly_duration": (
                    diagnosis.duration.model_dump() if diagnosis.duration else None
                ),
            })
        else:
            response.update({
                "root_cause": None,
                "confidence": None,
                "reasoning": None,
                "topology_evidence": None,
                "ml_top3": None,
                "anomaly_duration": None,
            })

        if evaluation:
            response["eval_passed"] = evaluation.passed
            response["eval_checks"] = evaluation.checks.model_dump()
            response["eval_retry_count"] = evaluation.retry_count

        response["latency_ms"] = round(elapsed_ms, 2)

        return response
