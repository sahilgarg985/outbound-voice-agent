from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""
    sip_outbound_trunk_id: str = ""
    agent_name: str = "healthcare-outbound-agent"

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen3:8b"
    llm_reasoning_effort: str | None = None

    def llm_options(self) -> dict:
        return {"reasoning_effort": self.llm_reasoning_effort} if self.llm_reasoning_effort else {}

    stt_base_url: str = "http://localhost:8001/v1"
    stt_api_key: str = "local"
    stt_model: str = "small.en"
    tts_base_url: str = "http://localhost:8001/v1"
    tts_api_key: str = "local"
    tts_model: str = "kokoro"
    tts_voice: str = "af_heart"

    max_call_seconds: int = 420
    clinic_name: str = "Riverside Health Clinic"
    agent_persona: str = "Maya"
    silence_timeout_s: float = 15.0

    data_dir: Path = PROJECT_ROOT / "data"
    recordings_dir: Path = PROJECT_ROOT / "recordings"
    scheduler_db: Path = PROJECT_ROOT / "data" / "appointments.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
