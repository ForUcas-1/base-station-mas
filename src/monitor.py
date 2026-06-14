"""Timed monitoring module — background asyncio task.

Runs the diagnostic pipeline on anomaly samples at fixed intervals.
Pushes real-time events via an asyncio Queue (consumed by SSE endpoint).
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from contracts.task import OrchestratorTask
from src.data.loader import DatasetLoader


class MonitorState:
    """Shared state for the monitoring system."""

    def __init__(self):
        self.running = False
        self.session_id: str | None = None
        self.current_sample_index: int | None = None
        self.current_sample_type: str = "normal"  # "normal" | "anomaly"
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._cancel_event: asyncio.Event = asyncio.Event()  # internal

    def push_event(self, event_type: str, data: dict[str, Any]):
        """Push an SSE event to all listeners."""
        data["type"] = event_type
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.event_queue.put_nowait(data)

    async def start(
        self,
        supervisor,
        interval_sec: float = 15.0,
        sample_index: int | None = None,
        sample_type: str = "anomaly",
    ):
        """Start timed monitoring."""
        if self.running:
            return

        self.running = True
        self.supervisor = supervisor  # 保存引用，stop 时需要
        self.session_id = str(uuid4())
        self.current_sample_index = sample_index
        self.current_sample_type = sample_type
        self._task = asyncio.create_task(
            self._monitor_loop(supervisor, interval_sec, sample_index, sample_type)
        )

    async def _monitor_loop(
        self, supervisor, interval_sec: float,
        _fixed_sample_index: int | None = None,
        _fixed_sample_type: str = "anomaly",
    ):
        """Main monitoring loop.

        Reads self.current_sample_index each round — supports
        user switching samples mid-monitoring via the UI.
        """
        loader = DatasetLoader()
        check_num = 0

        from src.db.database import insert_session, update_session
        await insert_session({
            "session_id": self.session_id,
            "interval_sec": interval_sec,
        })

        self.push_event("monitor_started", {
            "session_id": self.session_id,
            "interval_sec": interval_sec,
        })

        while self.running:
            check_num += 1
            t_start = time.perf_counter()

            # Read current sample (may have changed since last round via UI)
            sample_idx = self.current_sample_index
            if sample_idx is None:
                sample_idx = _pick_random_anomaly(loader)
                self.current_sample_index = sample_idx
                self.current_sample_type = "anomaly"

            sample = loader[sample_idx]
            anomaly_type = sample.get("anomalies", {}).get("type", "Unknown")

            # Lightweight sample_loaded event (no KPI clear — keep last round's data)
            self.push_event("monitor_tick", {
                "check_num": check_num,
                "sample_index": sample_idx,
                "anomaly_type": anomaly_type,
            })

            # Build task
            task = OrchestratorTask(
                user_query=f"[定时监测 #{check_num}] 样本 #{sample_idx} 异常检测",
                intent="diagnose",
                sample_index=sample_idx,
            )

            # SSE callback: push detection result immediately when ready
            def on_event(evt_type, data):
                if evt_type == "detection_done":
                    self.push_event("detection_done", {
                        "check_num": check_num,
                        **data,
                    })

            # Run pipeline (detection pushes event immediately via callback)
            result = await supervisor.handle_task(task, on_event=on_event)

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            has_anomaly = result.get("has_anomaly", False)

            # Store to DB
            from src.db.database import insert_diagnosis
            await insert_diagnosis({
                "run_id": result.get("task_id", str(uuid4())),
                "mode": f"timed_{(anomaly_type or 'anomaly').lower()}",
                "sample_index": sample_idx,
                "sample_type": self.current_sample_type,
                "anomaly_type": anomaly_type,
                "user_query": task.user_query,
                "has_anomaly": has_anomaly,
                "anomaly_score": result.get("anomaly_score", 0),
                "affected_kpis": result.get("affected_kpis", []),
                "root_cause": result.get("root_cause"),
                "confidence": result.get("confidence"),
                "reasoning": result.get("reasoning"),
                "topology_evidence": result.get("topology_evidence", []),
                "report_markdown": result.get("report_markdown"),
                "eval_passed": result.get("eval_passed", False),
                "latency_ms": elapsed_ms,
            })

            self.push_event("round_complete", {
                "check_num": check_num,
                "has_anomaly": has_anomaly,
                "anomaly_score": result.get("anomaly_score"),
                "affected_kpis": result.get("affected_kpis", []),
                "root_cause": result.get("root_cause"),
                "confidence": result.get("confidence"),
                "topology_evidence": result.get("topology_evidence", []),
                "report_markdown": result.get("report_markdown"),
                "latency_ms": round(elapsed_ms, 1),
            })

            # Update session counters
            await update_session(self.session_id, {
                "total_checks": check_num,
                "anomalies_found": (getattr(
                    await _get_session_counter(self.session_id),
                    "anomalies_found", 0
                ) or 0) + (1 if has_anomaly else 0),
                "reports_generated": (getattr(
                    await _get_session_counter(self.session_id),
                    "reports_generated", 0
                ) or 0) + 1,
            })

            # ── [异常自动停止] 正常走完当前报告 → 优雅退出 ──
            if has_anomaly:
                self.push_event("monitor_stopped", {
                    "reason": f"✅ 第 {check_num} 轮检测到异常，报告已生成，自动停止"
                })
                self.running = False
                await update_session(self.session_id, {"status": "stopped"})
                break

            # Wait for next interval
            await asyncio.sleep(interval_sec)

    async def stop(self):
        """[手动停止] 强制中断 — 关闭 LLM HTTP 连接 + cancel task。"""
        self.running = False
        # 关闭 LLM 连接 — 正在跑的 LLM 请求会立即断开
        if hasattr(self, 'supervisor'):
            await self.supervisor.stop_all()
        # Cancel 后台任务
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.push_event("monitor_stopped", {
            "reason": "⏹ 手动停止 — 已强制中断所有 Agent"
        })


# Global singleton
monitor_state = MonitorState()


async def handle_monitor_start(
    supervisor,
    interval_sec: float = 15.0,
    sample_index: int | None = None,
    sample_type: str = "anomaly",
) -> dict:
    """API handler: start monitoring."""
    await monitor_state.start(supervisor, interval_sec, sample_index, sample_type)
    return {"status": "started", "session_id": monitor_state.session_id}


async def handle_monitor_stop() -> dict:
    """API handler: stop monitoring."""
    await monitor_state.stop()
    return {"status": "stopped"}


async def handle_monitor_status() -> dict:
    """API handler: get current monitor status."""
    return {
        "running": monitor_state.running,
        "session_id": monitor_state.session_id,
        "current_sample": monitor_state.current_sample_index,
    }


def _pick_random_anomaly(loader: DatasetLoader) -> int:
    """Pick a random anomaly sample from anywhere in the dataset."""
    import random
    # Randomly sample across the full dataset
    for _ in range(100):
        i = random.randint(0, len(loader) - 1)
        s = loader[i]
        t = s.get("anomalies", {}).get("type", "")
        if t and t != "None":
            return i
    # Fallback: scan first 10000
    candidates = []
    for i in range(min(len(loader), 10000)):
        s = loader[i]
        t = s.get("anomalies", {}).get("type", "")
        if t and t != "None":
            candidates.append(i)
        if len(candidates) >= 50:
            break
    return random.choice(candidates) if candidates else 100


def _pick_random_normal(loader: DatasetLoader) -> int:
    """Pick a random normal sample index."""
    import random
    candidates = []
    limit = min(len(loader), 5000)
    for i in range(limit):
        s = loader[i]
        a = s.get("anomalies", {})
        if not a or not a.get("type"):
            candidates.append(i)
        if len(candidates) >= 50:
            break
    return random.choice(candidates) if candidates else 0


async def _get_session_counter(session_id: str):
    """Internal helper — get session counters for update."""
    from src.db.database import DB_PATH
    import aiosqlite
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT anomalies_found, reports_generated FROM monitoring_sessions WHERE session_id=?",
            (session_id,),
        )
        return await cursor.fetchone()
