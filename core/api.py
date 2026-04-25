from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .queue import MutationQueue
from .records import RunRecord
from .store import EvolutionStore


class RegisterRunRequest(BaseModel):
    run: dict


def create_app(arena_root: str | Path) -> FastAPI:
    store = EvolutionStore(arena_root)
    queue = MutationQueue(arena_root)
    app = FastAPI(title="Stable Evolution Arena")

    @app.get("/leaderboard")
    async def get_leaderboard(limit: int = 10):
        return store.leaderboard(limit)

    @app.get("/pareto")
    async def get_pareto():
        return store.pareto_front()

    @app.get("/queue")
    async def get_queue(status: str | None = None):
        return queue.list(status)

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str):
        row = store.get(run_id)
        if not row:
            raise HTTPException(status_code=404, detail="run not found")
        return row

    @app.get("/runs/{run_id}/lineage")
    async def get_lineage(run_id: str):
        return {"run_id": run_id, "lineage": store.lineage(run_id)}

    @app.get("/runs/{run_id}/children")
    async def get_children(run_id: str):
        return {"run_id": run_id, "children": store.children(run_id)}

    @app.get("/search")
    async def search(q: str, limit: int = 10):
        return store.search(q, limit=limit)

    @app.post("/runs/register")
    async def register(req: RegisterRunRequest):
        return store.register(RunRecord(**req.run))

    return app
