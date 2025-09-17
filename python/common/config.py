import os
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv


# Load .env for LOCAL runs only. In Docker, env vars are set in docker-compose.yml
if not os.getenv("CONTAINERIZED"):
    env_path = find_dotenv(usecwd=True) or (os.path.dirname(os.path.abspath(__file__)) + "/.env")
    load_dotenv(env_path, override=False)
    
def _require(name: str) -> str:
    """Helper to require env var and raise if not found."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required env var: {name}")
    return value

@dataclass(frozen=True)
class Settings:
    """Configuration settings loaded from environment variables."""

    # Twelve Data API key
    TD_KEY: str = _require("TWELVE_DATA_KEY")
    
settings = Settings()