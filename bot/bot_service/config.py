import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    bot_token: str
    rag_url: str
    postgres_dsn: str


def load_settings() -> Settings:
    return Settings(
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        rag_url=os.getenv("RAG_URL", "http://rag:8000"),
        postgres_dsn=os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@postgres:5432/rag"),
    )

