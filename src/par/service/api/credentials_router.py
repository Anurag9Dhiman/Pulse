from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from par.service.api.schemas import (
    LLMCredentialSet,
    LLMCredentialStatus,
    LLMProvider,
    RobotCreate,
    RobotCreated,
    RobotInfo,
)
from par.service.auth import ROBOT_TOKEN_PREFIX, generate_token, get_current_tenant, get_db, hash_token
from par.service.db.models import LLMCredential, RobotCredential, Tenant

router = APIRouter(prefix="/v1")


def _llm_credential(db: Session, tenant: Tenant, provider: str) -> LLMCredential | None:
    return db.scalar(
        select(LLMCredential).where(LLMCredential.tenant_id == tenant.id, LLMCredential.provider == provider)
    )


def _robot_credential(db: Session, tenant: Tenant, robot_id: str) -> RobotCredential | None:
    return db.scalar(
        select(RobotCredential).where(
            RobotCredential.tenant_id == tenant.id, RobotCredential.robot_id == robot_id
        )
    )


@router.put("/llm-credentials/{provider}", response_model=LLMCredentialStatus)
def set_llm_credential(
    provider: LLMProvider,
    body: LLMCredentialSet,
    request: Request,
    tenant: Tenant = Depends(get_current_tenant),
    db: Session = Depends(get_db),
) -> LLMCredentialStatus:
    encrypted = request.app.state.cipher.encrypt(body.api_key)
    credential = _llm_credential(db, tenant, provider)
    if credential is None:
        db.add(LLMCredential(tenant_id=tenant.id, provider=provider, encrypted_api_key=encrypted))
    else:
        credential.encrypted_api_key = encrypted
    db.commit()
    return LLMCredentialStatus(provider=provider, configured=True)


@router.delete("/llm-credentials/{provider}", status_code=204)
def delete_llm_credential(
    provider: LLMProvider, tenant: Tenant = Depends(get_current_tenant), db: Session = Depends(get_db)
) -> Response:
    credential = _llm_credential(db, tenant, provider)
    if credential is None:
        raise HTTPException(status_code=404, detail="no credential for this provider")
    db.delete(credential)
    db.commit()
    return Response(status_code=204)


@router.post("/robots", status_code=201, response_model=RobotCreated)
def register_robot(
    body: RobotCreate, tenant: Tenant = Depends(get_current_tenant), db: Session = Depends(get_db)
) -> RobotCreated:
    if _robot_credential(db, tenant, body.robot_id) is not None:
        raise HTTPException(status_code=409, detail="robot_id already registered")
    token = generate_token(ROBOT_TOKEN_PREFIX)
    db.add(RobotCredential(tenant_id=tenant.id, robot_id=body.robot_id, key_hash=hash_token(token)))
    db.commit()
    return RobotCreated(robot_id=body.robot_id, robot_token=token)


@router.get("/robots", response_model=list[RobotInfo])
def list_robots(
    request: Request, tenant: Tenant = Depends(get_current_tenant), db: Session = Depends(get_db)
) -> list[RobotInfo]:
    connections = request.app.state.connections
    credentials = db.scalars(
        select(RobotCredential).where(RobotCredential.tenant_id == tenant.id).order_by(RobotCredential.robot_id)
    )
    return [
        RobotInfo(
            robot_id=c.robot_id,
            connected=connections.get(str(tenant.id), c.robot_id) is not None,
            created_at=c.created_at,
        )
        for c in credentials
    ]


@router.delete("/robots/{robot_id}", status_code=204)
def delete_robot(
    robot_id: str, tenant: Tenant = Depends(get_current_tenant), db: Session = Depends(get_db)
) -> Response:
    credential = _robot_credential(db, tenant, robot_id)
    if credential is None:
        raise HTTPException(status_code=404, detail="unknown robot_id")
    db.delete(credential)
    db.commit()
    return Response(status_code=204)
