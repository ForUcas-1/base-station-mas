"""BaseStation-MAS — 5G Base Station Alarm Intelligent Diagnosis System.

Multi-Agent system using Supervisor topology + Evaluator quality loop.

Usage:
    python src/main.py

Environment:
    cp .env.example .env   # Edit with your API keys
    bash setup_env.sh      # Create venv + install deps
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure src/ is on sys.path so that 'from encoders.X import Model' works
# (the copied TelecomTS encoders use absolute imports like
#  'from encoders.utils.layers.Embed import DataEmbedding')
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Also add project root for knowledge_graph/ and configs/ relative paths
_PROJECT_ROOT = _SRC_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from src.utils.logging import get_logger

load_dotenv()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + Supervisor. Shutdown: cleanup."""
    from src.orchestrator.supervisor import Supervisor
    from src.db.database import init_db

    logger.info("Initializing SQLite database...")
    db_path = await init_db()
    logger.info(f"Database ready: {db_path}")

    logger.info("Initializing BaseStation-MAS Supervisor...")
    config_path = os.environ.get(
        "MAS_CONFIG",
        str(_PROJECT_ROOT / "configs" / "default.yaml"),
    )
    app.state.supervisor = Supervisor(config_path)
    logger.info("Supervisor ready ✓")
    logger.info(f"Dashboard: http://0.0.0.0:8000")
    yield
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create the FastAPI application with all routes."""
    app = FastAPI(
        title="BaseStation-MAS",
        description="5G 基站告警智能诊断系统 — Multi-Agent Architecture",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static files (vis-network, etc.) ──
    from fastapi.staticfiles import StaticFiles
    web_dir = os.path.join(os.path.dirname(__file__), "web")
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    # ── Web UI routes (dashboard, SSE, monitor, etc.) ──
    from src.web.routes import router as web_router
    app.include_router(web_router)

    # ── Legacy API routes (curl-compatible) ──
    @app.post("/diagnose")
    async def diagnose(request: Request):
        """Legacy endpoint: natural language diagnostic query."""
        body = await request.json()
        query = body.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="Field 'query' is required")

        supervisor: object = app.state.supervisor
        result = await supervisor.handle_query(query)
        return result

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        supervisor: object = app.state.supervisor
        return supervisor.worker_status()

    return app


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
app = create_app()

if __name__ == "__main__":
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    log_level = os.environ.get("LOG_LEVEL", "info")

    print(f"\n{'='*60}")
    print(f"  BaseStation-MAS v0.1.0")
    print(f"  5G 基站告警智能诊断系统")
    print(f"  Dashboard: http://{host}:{port}")
    print(f"{'='*60}\n")

    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
    )
