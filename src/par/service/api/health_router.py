from __future__ import annotations

from fastapi import APIRouter, Request, Response
from sqlalchemy import text

router = APIRouter(prefix="/v1")


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request, response: Response) -> dict[str, str]:
    try:
        with request.app.state.session_factory() as session:
            session.execute(text("SELECT 1"))
    except Exception:
        response.status_code = 503
        return {"status": "database unavailable"}
    return {"status": "ok"}
