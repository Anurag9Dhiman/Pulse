from __future__ import annotations

from dotenv import find_dotenv, load_dotenv


def load_env() -> None:
    """Loads a project .env if one exists, without overriding variables already set."""
    load_dotenv(find_dotenv(usecwd=True), override=False)
