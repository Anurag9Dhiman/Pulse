from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI

from par.core.planner import Planner
from par.service.api import admin_router, credentials_router, health_router, robots_router, tasks_router
from par.service.config import ServiceSettings
from par.service.crypto import SecretCipher
from par.service.db.base import make_engine, make_session_factory
from par.service.robots.connection_registry import RobotConnectionRegistry
from par.service.tasks.registry import TaskRegistry

PlannerFactory = Callable[[str, str | None], Planner]


def create_app(
    settings: ServiceSettings | None = None, planner_factory: PlannerFactory | None = None
) -> FastAPI:
    settings = settings or ServiceSettings()
    app = FastAPI(title="PAR service", version="1")

    app.state.settings = settings
    app.state.cipher = SecretCipher(settings.master_key)
    app.state.session_factory = make_session_factory(make_engine(settings.database_url))
    app.state.planner_factory = planner_factory or tasks_router.build_planner
    # Both registries are per-process: run exactly one worker/replica.
    app.state.tasks = TaskRegistry()
    app.state.connections = RobotConnectionRegistry()

    for module in (health_router, admin_router, credentials_router, tasks_router, robots_router):
        app.include_router(module.router)
    return app
