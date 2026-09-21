"""Shared project location and optional environment settings."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_environment():
    """Load the repository's .env without overwriting shell settings."""
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
