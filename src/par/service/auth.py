from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from par.service.db.models import ApiKey, RobotCredential, Tenant

API_KEY_PREFIX = "par_"
ROBOT_TOKEN_PREFIX = "parrobot_"


def generate_token(prefix: str) -> str:
    return prefix + secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """Unsalted SHA-256 is deliberate: tokens are 256 bits of randomness, not
    guessable passwords, and the hash doubles as the lookup key."""
    return hashlib.sha256(token.encode()).hexdigest()


def get_db(request: Request) -> Iterator[Session]:
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_current_tenant(
    authorization: str | None = Header(default=None), db: Session = Depends(get_db)
) -> Tenant:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="missing bearer token")
    api_key = db.scalar(
        select(ApiKey).where(ApiKey.key_hash == hash_token(token), ApiKey.revoked_at.is_(None))
    )
    if api_key is None:
        raise HTTPException(status_code=401, detail="invalid or revoked API key")
    tenant = db.get(Tenant, api_key.tenant_id)
    if tenant is None:
        raise HTTPException(status_code=401, detail="invalid or revoked API key")
    return tenant


def require_admin_token(request: Request, x_admin_token: str | None = Header(default=None)) -> None:
    expected = request.app.state.settings.admin_token
    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="invalid admin token")


@dataclass(frozen=True)
class RobotIdentity:
    tenant_id: uuid.UUID
    robot_id: str


def authenticate_robot_token(session_factory: sessionmaker[Session], token: str) -> RobotIdentity | None:
    """The robot's identity comes from the credential row, never from anything the client claims."""
    if not token:
        return None
    with session_factory() as session:
        credential = session.scalar(
            select(RobotCredential).where(RobotCredential.key_hash == hash_token(token))
        )
        if credential is None:
            return None
        return RobotIdentity(tenant_id=credential.tenant_id, robot_id=credential.robot_id)
