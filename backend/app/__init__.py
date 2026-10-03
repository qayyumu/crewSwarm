"""Load repo-root .env into the process before anything reads env vars."""

from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # dotenv is optional; plain env vars still work
    pass
