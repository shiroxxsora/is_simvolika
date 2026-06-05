import asyncio
import logging

from aiogram import Bot, Dispatcher

from bot_service.config import load_settings
from bot_service.handlers import register_handlers
from bot_service.profile_repository import ProfileRepository
from bot_service.rag_client import RAGClient


logging.basicConfig(level=logging.INFO)


async def run() -> None:
    settings = load_settings()
    if not settings.bot_token:
        raise RuntimeError("Environment variable TELEGRAM_BOT_TOKEN is required")

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher()
    rag_client = RAGClient(settings.rag_url)
    profile_repo = ProfileRepository(settings.postgres_dsn)

    register_handlers(
        dp=dp,
        bot=bot,
        rag_client=rag_client,
        rag_url=settings.rag_url,
        profile_repo=profile_repo,
    )
    await dp.start_polling(bot)


def main() -> None:
    asyncio.run(run())

