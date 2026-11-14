from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from par.service.api.schemas import ApiKeyCreate, ApiKeyCreated, TenantCreate, TenantCreated
from par.service.auth import API_KEY_PREFIX, generate_token, get_db, hash_token, require_admin_token
from par.service.db.models import ApiKey, Tenant

router = APIRouter(prefix="/v1/admin", dependencies=[Depends(require_admin_token)])


@router.post("/tenants", status_code=201, response_model=TenantCreated)
def create_tenant(body: TenantCreate, db: Session = Depends(get_db)) -> TenantCreated:
    tenant = Tenant(name=body.name)
    db.add(tenant)
    db.commit()
    return TenantCreated(tenant_id=str(tenant.id))


@router.post("/tenants/{tenant_id}/api-keys", status_code=201, response_model=ApiKeyCreated)
def create_api_key(tenant_id: uuid.UUID, body: ApiKeyCreate, db: Session = Depends(get_db)) -> ApiKeyCreated:
    if db.get(Tenant, tenant_id) is None:
        raise HTTPException(status_code=404, detail="unknown tenant")
    token = generate_token(API_KEY_PREFIX)
    key_prefix = token[:12]
    db.add(ApiKey(tenant_id=tenant_id, key_prefix=key_prefix, key_hash=hash_token(token), label=body.label))
    db.commit()
    return ApiKeyCreated(api_key=token, key_prefix=key_prefix)
