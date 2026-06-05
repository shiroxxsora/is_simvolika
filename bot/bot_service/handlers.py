import asyncio
import contextlib
import io

from aiogram import Bot, Dispatcher
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram import F
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

from bot_service.image_prepare import (
    guess_mime_for_document,
    guess_mime_for_telegram_photo,
    maybe_downscale_image,
)
from bot_service.profile_repository import DEFAULT_DISPLAY_NAME, ProfileRepository
from bot_service.rag_client import RAGClient


MAX_TELEGRAM_MESSAGE_LEN = 4000


def _main_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="⚙️ Настройки"), KeyboardButton(text="👤 Профиль")],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
        selective=False,
    )


def _split_message(text: str, max_len: int = MAX_TELEGRAM_MESSAGE_LEN) -> list[str]:
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n", 0, max_len)
        if split_at <= 0:
            split_at = max_len
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return chunks


async def _answer_long(message: Message, text: str) -> None:
    for part in _split_message(text):
        await message.answer(part)


async def _pulse_upload_photo(bot: Bot, chat_id: int) -> None:
    try:
        while True:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
            await asyncio.sleep(4.0)
    except asyncio.CancelledError:
        return


def _parse_command_arg(text: str | None) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    parts = s.split(maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()

def _parse_kv_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k and v:
            out[k] = v
    return out


def _settings_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        ("ФИО", "profile:set:full_name"),
        ("Основание (письмо/запрос)", "profile:set:basis"),
        ("Должность", "profile:set:position"),
        ("Образование", "profile:set:education"),
        ("Квалификация", "profile:set:qualification"),
        ("Повышение квалификации", "profile:set:training"),
        ("Интересы", "profile:set:interests"),
        ("Стаж", "profile:set:experience"),
        ("Показать профиль", "profile:show"),
        ("Отмена", "profile:cancel"),
    ]
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=label, callback_data=cb)]
            for label, cb in buttons
        ]
    )


def register_handlers(
    dp: Dispatcher,
    bot: Bot,
    rag_client: RAGClient,
    rag_url: str,
    profile_repo: ProfileRepository,
) -> None:
    image_processing_lock = asyncio.Lock()
    pending_profile_field: dict[int, str] = {}

    def _clear_pending(user_id: int | None) -> None:
        if user_id is None:
            return
        pending_profile_field.pop(user_id, None)

    @dp.message(CommandStart())
    async def start_handler(message: Message) -> None:
        _clear_pending(message.from_user.id if message.from_user else None)
        await message.answer(
            "ИС Символика\n\n"
            "Отправьте вопрос текстом или пришлите фото — подготовлю результат.\n"
            "Профиль специалиста — в меню «Профиль/Настройки».",
            reply_markup=_main_menu_keyboard(),
        )

    @dp.message(F.text == "⚙️ Настройки")
    async def settings_button_handler(message: Message) -> None:
        await settings_handler(message)

    @dp.message(F.text == "👤 Профиль")
    async def profile_button_handler(message: Message) -> None:
        await profile_handler(message)

    @dp.message(Command("profile"))
    async def profile_handler(message: Message) -> None:
        if not message.from_user:
            await message.answer("Не удалось определить пользователя Telegram.")
            return
        _clear_pending(message.from_user.id)
        profile = await asyncio.to_thread(profile_repo.get_profile, message.from_user.id)
        lines = ["Ваш профиль:"]
        lines.append(f"ФИО: {(profile.full_name or '').strip() or DEFAULT_DISPLAY_NAME}")
        if (profile.specialist_position or "").strip():
            lines.append(f"Должность: {profile.specialist_position}")
        if (profile.specialist_qualification or "").strip():
            lines.append(f"Квалификация: {profile.specialist_qualification}")
        if (profile.specialist_education or "").strip():
            lines.append(f"Образование: {profile.specialist_education}")
        if (profile.specialist_additional_training or "").strip():
            lines.append(f"Повышение квалификации: {profile.specialist_additional_training}")
        if (profile.specialist_research_interests or "").strip():
            lines.append(f"Интересы: {profile.specialist_research_interests}")
        if (profile.specialist_experience_years or "").strip():
            lines.append(f"Стаж: {profile.specialist_experience_years}")
        if (profile.report_basis or "").strip():
            lines.append(f"Основание (письмо/запрос): {profile.report_basis}")
        lines.append("")
        lines.append("Заполнение специалиста: /spec (в сообщении строки вида key=value).")
        lines.append("Поддерживаемые ключи: education, qualification, training, position, interests, experience, basis")
        await message.answer("\n".join(lines))

    @dp.message(Command("settings"))
    async def settings_handler(message: Message) -> None:
        _clear_pending(message.from_user.id if message.from_user else None)
        await message.answer(
            "Настройки профиля. Выберите поле и отправьте значение следующим сообщением.",
            reply_markup=_settings_keyboard(),
        )

    @dp.callback_query(F.data.startswith("profile:"))
    async def settings_callback(query: CallbackQuery) -> None:
        if not query.from_user:
            return
        data = (query.data or "").strip()
        if not data:
            return
        if data == "profile:cancel":
            pending_profile_field.pop(query.from_user.id, None)
            await query.message.answer("Отменено.")
            await query.answer()
            return
        if data == "profile:show":
            profile = await asyncio.to_thread(profile_repo.get_profile, query.from_user.id)
            lines = ["Ваш профиль:"]
            lines.append(f"ФИО: {(profile.full_name or '').strip() or DEFAULT_DISPLAY_NAME}")
            if (profile.specialist_position or "").strip():
                lines.append(f"Должность: {profile.specialist_position}")
            if (profile.specialist_qualification or "").strip():
                lines.append(f"Квалификация: {profile.specialist_qualification}")
            if (profile.specialist_education or "").strip():
                lines.append(f"Образование: {profile.specialist_education}")
            if (profile.specialist_experience_years or "").strip():
                lines.append(f"Стаж: {profile.specialist_experience_years}")
            if (profile.report_basis or "").strip():
                lines.append(f"Основание (письмо/запрос): {profile.report_basis}")
            await query.message.answer("\n".join(lines))
            await query.answer()
            return
        if data.startswith("profile:set:"):
            field = data.split(":", 2)[-1].strip()
            allowed = {
                "full_name",
                "basis",
                "position",
                "education",
                "qualification",
                "training",
                "interests",
                "experience",
            }
            if field not in allowed:
                await query.answer()
                return
            pending_profile_field[query.from_user.id] = field
            pretty = {
                "full_name": "ФИО",
                "basis": "Основание (письмо/запрос)",
                "position": "Должность",
                "education": "Образование",
                "qualification": "Квалификация",
                "training": "Повышение квалификации",
                "interests": "Интересы",
                "experience": "Стаж",
            }.get(field, field)
            await query.message.answer(f"Введите значение для поля: {pretty}")
            await query.answer()
            return
        await query.answer()

    @dp.message(Command("fio"))
    async def fio_handler(message: Message) -> None:
        if not message.from_user:
            await message.answer("Не удалось определить пользователя Telegram.")
            return
        _clear_pending(message.from_user.id)
        fio = _parse_command_arg(message.text)
        if not fio:
            await message.answer("Укажите ФИО после команды. Пример:\n/fio Иванов Иван Иванович")
            return
        profile, changed = await asyncio.to_thread(profile_repo.upsert_full_name, message.from_user.id, fio)
        if changed:
            await message.answer(f"ФИО сохранено: {profile.full_name}")
        else:
            await message.answer(f"ФИО уже установлено: {profile.display_name}")

    @dp.message(Command("spec"))
    async def spec_handler(message: Message) -> None:
        if not message.from_user:
            await message.answer("Не удалось определить пользователя Telegram.")
            return
        _clear_pending(message.from_user.id)
        body = _parse_command_arg(message.text)
        if not body:
            await message.answer(
                "Отправьте /spec и далее многострочно key=value.\n"
                "Пример:\n"
                "/spec\n"
                "position=Старший преподаватель ...\n"
                "qualification=юрист ...\n"
                "education=высшее ...\n"
                "training=удостоверение ...\n"
                "interests=...\n"
                "experience=15 лет\n"
                "basis=запрос ...\n"
            )
            return
        kv = _parse_kv_lines(body)
        profile, changed = await asyncio.to_thread(
            profile_repo.upsert_specialist_fields,
            message.from_user.id,
            education=kv.get("education"),
            qualification=kv.get("qualification"),
            additional_training=kv.get("training"),
            position=kv.get("position"),
            research_interests=kv.get("interests"),
            experience_years=kv.get("experience"),
            report_basis=kv.get("basis"),
        )
        if changed:
            await message.answer("Профиль обновлён.")
        else:
            await message.answer("Без изменений.")

    @dp.message()
    async def rag_chat_handler(message: Message) -> None:
        # If user is currently entering a profile field via /settings buttons, consume this message.
        if message.from_user and message.text:
            field = pending_profile_field.get(message.from_user.id)
            if field:
                value = (message.text or "").strip()
                pending_profile_field.pop(message.from_user.id, None)
                if not value:
                    await message.answer("Пустое значение. Выберите поле в «Настройки» и отправьте значение.")
                    return
                if field == "full_name":
                    profile, changed = await asyncio.to_thread(
                        profile_repo.upsert_full_name, message.from_user.id, value
                    )
                    if changed:
                        await message.answer(f"ФИО сохранено: {profile.full_name}")
                    else:
                        await message.answer(f"ФИО уже установлено: {profile.display_name}")
                    return
                kwargs = {}
                if field == "basis":
                    kwargs["report_basis"] = value
                elif field == "position":
                    kwargs["position"] = value
                elif field == "education":
                    kwargs["education"] = value
                elif field == "qualification":
                    kwargs["qualification"] = value
                elif field == "training":
                    kwargs["additional_training"] = value
                elif field == "interests":
                    kwargs["research_interests"] = value
                elif field == "experience":
                    kwargs["experience_years"] = value
                profile, changed = await asyncio.to_thread(
                    profile_repo.upsert_specialist_fields,
                    message.from_user.id,
                    **kwargs,
                )
                if changed:
                    await message.answer("Сохранено.")
                else:
                    await message.answer("Без изменений.")
                return

        file_id: str | None = None
        mime_guess = guess_mime_for_telegram_photo()

        if message.photo:
            file_id = message.photo[-1].file_id
            mime_guess = guess_mime_for_telegram_photo()
        elif message.document and guess_mime_for_document(message.document.mime_type):
            file_id = message.document.file_id
            mg = guess_mime_for_document(message.document.mime_type)
            mime_guess = mg or "image/jpeg"

        if file_id:
            if message.from_user:
                profile = await asyncio.to_thread(profile_repo.get_profile, message.from_user.id)
                if not (profile.full_name or "").strip():
                    await message.answer(
                        "Предупреждение: у вас не задано ФИО. "
                        "Задайте его командой /fio Иванов Иван Иванович — это нужно для профиля."
                    )

            if image_processing_lock.locked():
                await message.answer(
                    "Сервис сейчас занят обработкой предыдущего фото. "
                    "Попробуйте отправить изображение чуть позже."
                )
                return

            async with image_processing_lock:
                pulse = asyncio.create_task(_pulse_upload_photo(bot, message.chat.id))
                try:
                    file = await bot.get_file(file_id)
                    buffer = io.BytesIO()
                    await bot.download_file(file.file_path, destination=buffer)
                    raw_bytes = buffer.getvalue()
                    prepared, out_mime = maybe_downscale_image(raw_bytes, mime_type=mime_guess)
                    user_hint = (message.caption or "").strip()
                    specialist_profile = None
                    if message.from_user:
                        specialist_profile = await asyncio.to_thread(
                            profile_repo.get_profile,
                            message.from_user.id,
                        )

                    result = await rag_client.ask_image(
                        image_bytes=prepared,
                        mime_type=out_mime,
                        user_hint=user_hint,
                        specialist_profile=specialist_profile,
                    )
                finally:
                    pulse.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await pulse

                text_out, report_bytes, report_name = result
                if report_bytes:
                    # Telegram limit ~50MB; keep a safety margin
                    if len(report_bytes) <= 45 * 1024 * 1024:
                        try:
                            await message.answer_document(
                                BufferedInputFile(
                                    report_bytes,
                                    filename=report_name or "заключение.docx",
                                )
                            )
                            return
                        except Exception:
                            await message.answer("Не удалось отправить файл. Попробуйте ещё раз чуть позже.")
                            return
                    await message.answer(
                        "Файл получился слишком большим для Telegram. Попробуйте другое изображение."
                    )
                    return

                await message.answer("Не удалось сформировать файл-отчёт. Попробуйте ещё раз.")
                return

        text = (message.text or "").strip()
        if not text:
            await message.answer("Пришлите текст или фото (можно с подписью).")
            return

        answer = await rag_client.ask_text(text)
        await _answer_long(message, answer)
