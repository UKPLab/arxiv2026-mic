"""Load optional project settings without overwriting the shell environment."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_environment():
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
