from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    """Constructing this with any required variable unset raises, so the
    service refuses to start rather than running without a database, an
    encryption key, or an admin secret."""

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")

    database_url: str = Field(validation_alias="DATABASE_URL")
    master_key: str = Field(validation_alias="PAR_MASTER_KEY")
    admin_token: str = Field(validation_alias="PAR_ADMIN_TOKEN")
    robot_request_timeout_seconds: float = Field(default=30.0, validation_alias="PAR_ROBOT_REQUEST_TIMEOUT")
