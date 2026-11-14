from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config

_SERVICE_DIR = Path(__file__).resolve().parent.parent


def alembic_config(database_url: str) -> Config:
    config = Config(str(_SERVICE_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(_SERVICE_DIR / "db" / "migrations"))
    # Alembic's ini parser treats % as interpolation syntax.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def upgrade(database_url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(database_url), revision)


def downgrade(database_url: str, revision: str = "base") -> None:
    command.downgrade(alembic_config(database_url), revision)


if __name__ == "__main__":
    upgrade(os.environ["DATABASE_URL"])
