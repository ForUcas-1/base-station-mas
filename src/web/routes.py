"""FastAPI router — WebSocket + simplified REST."""

import asyncio
import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse

from src.data.loader import DatasetLoader
from src.monitor import (
    MonitorState, _pick_random_anomaly, _pick_random_normal,
    monitor_state as global_monitor,
)

router = APIRouter()
loader = DatasetLoader()
DASHBOARD_PATH = Path(__file__).resolve().parent / "dashboard.html"
TOPOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "knowledge_graph" / "topology.json"


def _resolve_topology_highlight(evidence, root_cause, zone):
    """Map root cause to SVG node/edge IDs for highlight."""
    from src.graphrag.graph import KnowledgeGraph
    kg = KnowledgeGraph(str(TOPOLOGY_PATH))
    rc_info = kg.query_by_root_cause(root_cause or "Unknown", f"Zone_{zone}")
    return rc_info.get("highlight_nodes", []), rc_info.get("highlight_edges", [])


# ═══════════════════════════════════════════════════════
# Page + Static
# ═══════════════════════════════════════════════════════
@router.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))


@router.get("/api/topology")
async def get_topology():
    if TOPOLOGY_PATH.exists():
        return json.loads(TOPOLOGY_PATH.read_text(encoding="utf-8"))
    return {"nodes": [], "edges": []}


# ═══════════════════════════════════════════════════════
# WebSocket — 统一通信
# ═══════════════════════════════════════════════════════
class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def send_callback(self, data: dict):
        """Sync callback — schedules broadcast on event loop."""
        asyncio.create_task(self.broadcast(data))


ws_manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    supervisor = ws.app.state.supervisor  # type: ignore[attr-defined]

    # Per-connection state
    cur_sample = None
    monitor = MonitorState()
    # Forward events from the global monitor's event loop to this WS
    # We use a dedicated monitor_state per connection but share supervisor

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            cmd = msg.get("cmd", "")

            if cmd == "select_sample":
                stype = msg.get("type", "anomaly")
                if stype == "anomaly":
                    # Model-assisted: try up to 20 candidates, keep first with score >= 0.5
                    sup = ws.app.state.supervisor
                    det = None
                    for _ in range(20):
                        idx = _pick_random_anomaly(loader)
                        s = loader[idx]
                        atype = s.get("anomalies", {}).get("type", "Unknown")
                        try:
                            from contracts.task import OrchestratorTask as OT
                            tt = OT(user_query="", intent="diagnose", sample_index=idx, subtasks=["detect"])
                            det = await sup.detection.execute(tt)
                            if det.anomaly_score >= 0.5:
                                break
                        except Exception:
                            pass
                    from src.utils.logging import get_logger
                    score = det.anomaly_score if det else 0
                    get_logger("ws").info(f"Selected anomaly sample #{idx}: {atype} (score={score:.3f})")
                else:
                    # Model-assisted: try up to 20 normal candidates, keep first with score < 0.5
                    sup = ws.app.state.supervisor
                    det = None
                    for _ in range(20):
                        idx = _pick_random_normal(loader)
                        try:
                            from contracts.task import OrchestratorTask as OT
                            tt = OT(user_query="", intent="diagnose", sample_index=idx, subtasks=["detect"])
                            det = await sup.detection.execute(tt)
                            if det.anomaly_score < 0.5:
                                break
                        except Exception:
                            pass
                    atype = ""
                cur_sample = {"index": idx, "type": stype, "anomaly_type": atype}
                monitor.current_sample_index = idx
                monitor.current_sample_type = stype
                await ws.send_json({
                    "type": "sample_loaded", "sample_index": idx,
                    "anomaly_type": atype,
                })

            elif cmd == "run_oneshot":
                if not cur_sample:
                    await ws.send_json({"type": "error", "message": "请先选择样本"})
                    continue

                await ws.send_json({"type": "status", "message": "oneshot_started"})
                await ws.send_json({"type": "agent_status", "agent": "Orchestrator", "status": "done"})

                # Step 1: Detection
                from contracts.task import OrchestratorTask
                await ws.send_json({"type": "agent_status", "agent": "Detection", "status": "running"})
                t = OrchestratorTask(
                    user_query=f"诊断样本 #{cur_sample['index']}",
                    intent="diagnose",
                    sample_index=cur_sample["index"],
                    subtasks=["detect"],
                )
                det = await supervisor.detection.execute(t)
                await ws.send_json({"type": "agent_status", "agent": "Detection", "status": "done"})
                await ws.send_json({
                    "type": "detection_done",
                    "has_anomaly": det.has_anomaly,
                    "anomaly_score": det.anomaly_score,
                    "affected_kpis": det.affected_kpis,
                    "inference_time_ms": det.inference_time_ms,
                })

                if not det.has_anomaly:
                    await ws.send_json({
                        "type": "round_complete", "has_anomaly": False,
                        "anomaly_score": det.anomaly_score,
                    })
                    continue

                # Step 2: Full diagnosis
                from contracts.detection import DetectionResult
                t2 = OrchestratorTask(
                    user_query=f"根因分析 #{cur_sample['index']}",
                    intent="diagnose",
                    sample_index=cur_sample["index"],
                    subtasks=["detect", "diagnose", "report"],
                )
                t2.context.detection_result = DetectionResult(
                    has_anomaly=True,
                    anomaly_score=det.anomaly_score,
                    affected_kpis=det.affected_kpis,
                    sample_index=cur_sample["index"],
                )

                t_start = time.perf_counter()
                result = await supervisor.handle_task(t2, t_start,
                    on_event=lambda evt, data: ws_manager.send_callback(
                        {"type": evt, **data} if isinstance(data, dict) else data))

                # Topology highlight
                evidence = result.get("topology_evidence", [])
                root_cause = result.get("root_cause", "")
                zone = "B"
                if evidence or root_cause:
                    hl_nodes, hl_edges = _resolve_topology_highlight(
                        evidence, root_cause, zone
                    )
                    await ws.send_json({
                        "type": "topology_highlight",
                        "nodes": hl_nodes, "edges": hl_edges,
                    })

                await ws.send_json({
                    "type": "round_complete",
                    "has_anomaly": True,
                    "anomaly_score": det.anomaly_score,
                    "affected_kpis": det.affected_kpis,
                    "root_cause": result.get("root_cause"),
                    "confidence": result.get("confidence"),
                    "reasoning": result.get("reasoning"),
                    "topology_evidence": result.get("topology_evidence", []),
                    "report_markdown": result.get("report_markdown"),
                    "latency_ms": result.get("latency_ms"),
                })

                # Save to DB
                from src.db.database import insert_diagnosis
                await insert_diagnosis({
                    "run_id": result.get("task_id", t2.task_id),
                    "mode": "oneshot",
                    "sample_index": cur_sample["index"],
                    "sample_type": cur_sample["type"],
                    "anomaly_type": cur_sample.get("anomaly_type"),
                    "user_query": f"诊断样本 #{cur_sample['index']}",
                    "has_anomaly": True,
                    "anomaly_score": det.anomaly_score,
                    "affected_kpis": det.affected_kpis,
                    "root_cause": result.get("root_cause"),
                    "confidence": result.get("confidence"),
                    "reasoning": result.get("reasoning"),
                    "topology_evidence": result.get("topology_evidence", []),
                    "report_markdown": result.get("report_markdown"),
                    "eval_passed": result.get("eval_passed", False),
                    "latency_ms": result.get("latency_ms", 0),
                })

            elif cmd == "start_monitor":
                if not cur_sample:
                    await ws.send_json({"type": "error", "message": "请先选择样本"})
                    continue
                interval = msg.get("interval_sec", 15)
                monitor.supervisor = supervisor
                monitor.running = True
                monitor.session_id = str(__import__('uuid').uuid4())
                monitor.current_sample_index = cur_sample["index"]
                monitor.current_sample_type = cur_sample["type"]
                await ws.send_json({
                    "type": "monitor_started",
                    "session_id": monitor.session_id,
                    "interval_sec": interval,
                })
                asyncio.create_task(
                    _monitor_ws_loop(monitor, supervisor, ws, interval)
                )

            elif cmd == "stop_monitor":
                monitor.running = False
                if hasattr(monitor, 'supervisor'):
                    await monitor.supervisor.stop_all()
                await ws.send_json({
                    "type": "monitor_stopped",
                    "reason": "⏹ 手动停止",
                })

    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(ws)


async def _monitor_ws_loop(
    monitor: MonitorState, supervisor, ws: WebSocket, interval_sec: float
):
    """Timed monitoring loop — pushes events via WebSocket."""
    from contracts.task import OrchestratorTask
    from src.db.database import insert_diagnosis, insert_session, update_session

    await insert_session({
        "session_id": monitor.session_id,
        "interval_sec": interval_sec,
    })

    check_num = 0
    while monitor.running:
        check_num += 1
        t_start = time.perf_counter()

        idx = monitor.current_sample_index
        if idx is None:
            idx = _pick_random_anomaly(loader)
        s = loader[idx]
        atype = s.get("anomalies", {}).get("type", "Unknown")

        await ws.send_json({
            "type": "monitor_tick",
            "check_num": check_num,
            "sample_index": idx,
            "anomaly_type": atype,
        })

        task = OrchestratorTask(
            user_query=f"[定时 #{check_num}] 样本 #{idx}",
            intent="diagnose", sample_index=idx,
            subtasks=["detect", "diagnose", "report"],
        )

        def on_event(evt_type, data):
            data["type"] = evt_type
            asyncio.create_task(ws.send_json(data))

        result = await supervisor.handle_task(task, on_event=on_event)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        has_anomaly = result.get("has_anomaly", False)

        # Topology
        root_cause = result.get("root_cause", "")
        if root_cause:
            hl_nodes, hl_edges = _resolve_topology_highlight(
                result.get("topology_evidence", []), root_cause, "B"
            )
            await ws.send_json({
                "type": "topology_highlight", "nodes": hl_nodes, "edges": hl_edges,
            })

        await ws.send_json({
            "type": "round_complete",
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

        # DB
        await insert_diagnosis({
            "run_id": result.get("task_id", task.task_id),
            "mode": f"timed_{atype.lower() if atype else 'anomaly'}",
            "sample_index": idx,
            "sample_type": monitor.current_sample_type,
            "anomaly_type": atype,
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

        if has_anomaly:
            await ws.send_json({
                "type": "monitor_stopped",
                "reason": f"✅ 第 {check_num} 轮检测到异常，报告已生成，自动停止",
            })
            monitor.running = False
            await update_session(monitor.session_id, {"status": "stopped"})
            break

        await asyncio.sleep(interval_sec)


# ═══════════════════════════════════════════════════════
# History
# ═══════════════════════════════════════════════════════
@router.delete("/api/history/{run_id}")
async def delete_run(run_id: str):
    """Delete a single diagnosis run."""
    from src.db.database import DB_PATH
    import aiosqlite
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("DELETE FROM diagnosis_runs WHERE run_id=?", (run_id,))
        await db.commit()
    return {"status": "deleted"}
@router.get("/api/history")
async def get_history(limit: int = 20):
    from src.db.database import get_history as db_history
    return await db_history(limit)


@router.get("/api/history/{run_id}")
async def get_run_detail(run_id: str):
    from src.db.database import get_run
    result = await get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return result
