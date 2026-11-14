from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError

from par.service.db.models import LLMCredential, RobotCredential, Tenant


def test_migration_creates_every_table_the_models_declare(database_url):
    # The app's tables come from the Alembic migration, not create_all(),
    # so a model added without a migration fails here.
    engine = create_engine(database_url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert {"tenants", "api_keys", "robot_credentials", "llm_credentials"} <= tables


def test_a_tenant_cannot_register_the_same_robot_id_twice(app):
    with app.state.session_factory() as session:
        tenant = Tenant(name="acme")
        session.add(tenant)
        session.flush()
        session.add(RobotCredential(tenant_id=tenant.id, robot_id="arm-1", key_hash="h1"))
        session.add(RobotCredential(tenant_id=tenant.id, robot_id="arm-1", key_hash="h2"))
        with pytest.raises(IntegrityError):
            session.commit()


def test_two_tenants_can_use_the_same_robot_id(app):
    with app.state.session_factory() as session:
        a, b = Tenant(name="a"), Tenant(name="b")
        session.add_all([a, b])
        session.flush()
        session.add(RobotCredential(tenant_id=a.id, robot_id="arm-1", key_hash="h1"))
        session.add(RobotCredential(tenant_id=b.id, robot_id="arm-1", key_hash="h2"))
        session.commit()


def test_a_tenant_has_at_most_one_credential_per_llm_provider(app):
    with app.state.session_factory() as session:
        tenant = Tenant(name="acme")
        session.add(tenant)
        session.flush()
        session.add(LLMCredential(tenant_id=tenant.id, provider="gemini", encrypted_api_key="x"))
        session.add(LLMCredential(tenant_id=tenant.id, provider="gemini", encrypted_api_key="y"))
        with pytest.raises(IntegrityError):
            session.commit()
