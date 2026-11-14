from __future__ import annotations

import os
from typing import Any

import pytest

from par.core.planner import Planner

try:
    import fastapi  # noqa: F401
    import sqlalchemy  # noqa: F401
except ImportError:  # the `service` extra isn't installed
    collect_ignore = ["test_api.py", "test_db.py"]


class ScriptedPlanner(Planner):
    def __init__(self, steps: list[tuple[str, dict[str, Any]]]) -> None:
        self._steps = list(steps)

    def propose(self, goal, observation, capabilities):
        return self._steps.pop(0)


class ScriptedPlannerFactory:
    """Stands in for build_planner so API tests never call a real LLM;
    records what the service passed so tests can check the tenant's own
    decrypted key is what reaches the planner."""

    def __init__(self) -> None:
        self.steps: list[tuple[str, dict[str, Any]]] = []
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, provider: str, api_key: str | None) -> Planner:
        self.calls.append((provider, api_key))
        return ScriptedPlanner(self.steps)


@pytest.fixture(scope="session")
def database_url():
    """DB-backed tests need a real Postgres (the engine production uses),
    so they skip rather than fall back to SQLite when none is configured."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    from par.service.db.migrate import downgrade, upgrade

    upgrade(url)
    yield url
    downgrade(url)


@pytest.fixture
def planner_factory() -> ScriptedPlannerFactory:
    return ScriptedPlannerFactory()


@pytest.fixture
def app(database_url, planner_factory):
    from sqlalchemy import text

    from par.service.app import create_app
    from par.service.config import ServiceSettings
    from par.service.crypto import generate_master_key

    settings = ServiceSettings(
        database_url=database_url,
        master_key=generate_master_key(),
        admin_token="test-admin-token",
        robot_request_timeout_seconds=5.0,
    )
    application = create_app(settings, planner_factory=planner_factory)
    yield application
    with application.state.session_factory() as session:
        session.execute(text("TRUNCATE tenants CASCADE"))
        session.commit()
    application.state.session_factory.kw["bind"].dispose()


@pytest.fixture
def client(app):
    from starlette.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client
