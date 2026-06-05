# ORMProject — RAG-система и Telegram-бот «ИС Символика»

Система экспертного анализа татуировок и символики на основе справочников. Состоит из RAG-сервиса, Telegram-бота и Docker-стека.

---

## Требования

| Компонент | Версия |
|---|---|
| Docker Desktop | ≥ 4.25 (с включённым Docker Compose v2) |
| LM Studio | последняя (если используете локальные модели) |
| Дисковое место | ≥ 10 ГБ (модели + база данных + PDF) |

---

## Шаг 1. Клонировать репозиторий

```bash
git clone <url-репозитория>
cd ORMProject
```

В репозитории уже включены:
- `rag/docs/` — исходные PDF
- `rag/docs_prepared/` — подготовленные TXT (результат OCR)
- `rag/docs_toc/` — файлы оглавления

---

## Шаг 2. Настроить `.env`

Скопируйте шаблон и заполните значения:

```bash
# Linux / macOS
cp docker/.env.example docker/.env

# Windows (cmd или PowerShell)
copy docker\.env.example docker\.env
```

Откройте `docker/.env` и заполните **обязательные** поля:

### Обязательные параметры

| Строка | Что вписать |
|---|---|
| `TELEGRAM_BOT_TOKEN=` | Токен бота от [@BotFather](https://t.me/BotFather) |
| `POSTGRES_PASSWORD=` | Пароль БД (замените `postgres` на надёжный) |
| `LLM_MODEL=` | Имя модели (см. ниже варианты) |
| `PREPARE_VL_MODEL=` | Та же VL-модель для OCR документов |
| `EMBEDDING_MODEL=` | Модель эмбеддингов |

> **Важно**: если поменяли `POSTGRES_USER` или `POSTGRES_PASSWORD` — обновите также строку `POSTGRES_DSN`:
> ```
> POSTGRES_DSN=postgresql://НОВЫЙ_ЮЗЕР:НОВЫЙ_ПАРОЛЬ@postgres:5432/rag
> ```

Файл `docker/.env` не коммитится в git. В репозитории хранится только `docker/.env.example`.

---

## Шаг 3. Выбрать источник моделей

### Вариант A: LM Studio (по умолчанию, бесплатно)

1. Запустите **LM Studio на хосте** — сервер должен работать **до** `docker compose up`.
2. Скачайте модели:
   - **Чат/RAG/Vision/OCR**: например `qwen/qwen3.5-9b`
   - **Эмбеддинги**: например `text-embedding-qwen3-embedding-8b`
3. В LM Studio → **Local Server** → нажмите **Start Server** (порт 1234).
4. Для каждой загруженной модели: вкладка модели → **Context Length** → установите **≥ 8192** (иначе длинные документы обрезаются).
5. В `docker/.env` убедитесь, что заданы:
   ```dotenv
   LLM_API_URL=http://host.docker.internal:1234/v1/chat/completions
   LLM_MODEL=qwen/qwen3.5-9b
   VISION_MODEL=qwen/qwen3.5-9b
   PREPARE_VL_API_URL=http://host.docker.internal:1234/v1/chat/completions
   PREPARE_VL_MODEL=qwen/qwen3.5-9b
   EMBEDDING_API_URL=http://host.docker.internal:1234/v1/embeddings
   EMBEDDING_MODEL=text-embedding-qwen3-embedding-8b
   EMBEDDING_DIM=4096
   ```

> **Linux / WSL**: если контейнер не достучивается до хоста (`Errno 101`), проверьте брандмауэр и что LM Studio слушает `0.0.0.0`, а не только `127.0.0.1`.

### Вариант B: OpenAI API

Замените следующие строки в `docker/.env`:

```dotenv
LLM_API_URL=https://api.openai.com/v1/chat/completions
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
VISION_MODEL=gpt-4o-mini

PREPARE_VL_API_URL=https://api.openai.com/v1/chat/completions
PREPARE_VL_MODEL=gpt-4o-mini

EMBEDDING_API_URL=https://api.openai.com/v1/embeddings
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

---

## Шаг 4. Документы и TOC-файлы

### 4.1 PDF-документы

Исходные PDF лежат в `rag/docs/`. При первом запуске контейнер `prepare_docs` обработает новые или изменённые файлы и положит результат в `rag/docs_prepared/`.

Если добавляете новый PDF:
1. Положите файл в `rag/docs/`
2. Создайте TOC-файл `rag/docs_toc/<имя>.txt` (см. ниже)
3. Пересоберите OCR и переиндексируйте RAG:

```bash
cd docker
docker compose run --rm prepare_docs
docker compose restart rag
```

### 4.2 TOC-файлы (оглавление — необязательно, но настоятельно рекомендуется)

TOC-файлы позволяют привязать каждую страницу PDF к названию главы — это улучшает качество ссылок в ответах RAG.

**Расположение:** `rag/docs_toc/`

**Имя файла:** `<имя_pdf_без_расширения>.txt` — расширение `.txt` обязательно.

Пример: для `MyBook.pdf` → `rag/docs_toc/MyBook.txt`

**Формат файла:**

```
source_doc: Полное человекочитаемое название книги / документа
source_chapter: Название первой главы
source_page: 5

[TOC]
5 | Вступление
12 | Глава 1. Название
35 | Глава 2. Название
78 | Заключение
```

| Строка | Описание |
|---|---|
| `source_doc:` | Название, которое будет показано в ответах бота как источник |
| `source_chapter:` | Глава для самой первой страницы (необязательно) |
| `source_page:` | Номер первой страницы документа |
| `[TOC]` | Маркер начала таблицы страниц — **обязателен** |
| `N \| Название` | Номер страницы PDF, с которой начинается глава |

**Правила:**
- Глава действует от своей страницы до начала следующей записи в TOC
- Если TOC-файла нет — документ всё равно индексируется, просто без глав
- Страницы нумеруются как в PDF (физические, не логические)

**Пример готового TOC** — см. `rag/docs_toc/Baldaev__Tatuirovki_zaklyuchennykh_2001.txt`

---

## Шаг 5. Запустить стек

**Перед запуском:** LM Studio (или другой LLM-провайдер) должен быть доступен — контейнер `prepare_docs` сразу обращается к VL API.

```bash
cd docker
docker compose up -d --build
```

Порядок запуска автоматический:

```
postgres → migrate → prepare_docs → rag → telegram_bot
```

Первый запуск занимает несколько минут (загрузка образов, OCR документов, индексация).

---

## Шаг 6. Проверить работу

```bash
cd docker

# Статус контейнеров
docker compose ps

# Логи RAG-сервиса
docker compose logs -f rag

# Логи бота
docker compose logs -f telegram_bot

# Health-check RAG
curl http://localhost:8000/health
```

Ответ `/health` содержит `status`, `indexing_done`, `rag_chunk_count` и др. Значение `"status": "degraded"` нормально, пока идёт индексация или в базе 0 чанков. После успешного старта ожидается `"status": "ok"` и `rag_chunk_count > 0`.

---

## Работа с Telegram-ботом

После запуска стека найдите бота в Telegram по имени, которое задали в @BotFather, и нажмите **Start** (или отправьте `/start`).

### Главное меню

Бот показывает клавиатуру с двумя кнопками:
- **⚙️ Настройки** — заполнение профиля специалиста по шагам
- **👤 Профиль** — просмотр сохранённых данных

### Текстовые вопросы

Отправьте обычное текстовое сообщение — бот ищет ответ в базе знаний и возвращает:
- ответ на вопрос;
- список источников с цитатами, страницами и оценкой релевантности.

Пример:
```
Что означает подключичная звезда?
```

### Анализ фото (основной сценарий)

1. Рекомендуется заполнить профиль специалиста (см. ниже) — данные подставляются в заключение.
2. Отправьте фото татуировки (как фото или как файл-изображение).
3. При необходимости добавьте **подпись** к фото — это подсказка для анализа (например, «звезда на груди»).
4. Бот пришлёт **только файл `.docx`** — готовое заключение. Текстовое описание и источники в чат не дублируются.

> Обработка одного фото занимает несколько минут. Пока идёт анализ, бот показывает статус «загружает фото». Второе фото нужно отправлять после завершения предыдущего.

> Если профиль не заполнен, в заключение подставляются значения по умолчанию. Бот предупредит, если ФИО пустое, но заключение всё равно сформирует.

### Профиль специалиста

Данные профиля используются при формировании заключения по фото.

**Способ 1 — кнопки «⚙️ Настройки»:**
1. Нажмите **⚙️ Настройки**
2. Выберите поле (ФИО, должность, образование и т.д.)
3. Отправьте значение следующим сообщением

**Способ 2 — команда `/fio`:**
```
/fio Иванов Иван Иванович
```

**Способ 3 — команда `/spec`** (все поля в **одном** сообщении):
```
/spec
position=Старший преподаватель кафедры ОРД
qualification=юрист по образованию
education=высшее юридическое
training=удостоверение о повышении квалификации
interests=криминология, оперативно-розыскная деятельность
experience=15 лет
basis=запрос начальника отдела
```

Поддерживаемые ключи для `/spec`:

| Ключ | Поле |
|---|---|
| `position` | Должность |
| `qualification` | Квалификация |
| `education` | Образование |
| `training` | Повышение квалификации |
| `interests` | Сфера научных интересов |
| `experience` | Стаж работы |
| `basis` | Основание (письмо/запрос) |

**Просмотр профиля:** кнопка **👤 Профиль** или команда `/profile`

### Команды бота

| Команда / действие | Что делает |
|---|---|
| `/start` | Приветствие и главное меню |
| Текстовое сообщение | Вопрос к базе знаний |
| Фото / файл-изображение | Анализ татуировки → только `.docx` |
| Подпись к фото | Дополнительная подсказка для анализа |
| `/fio Иванов И.И.` | Сохранить ФИО специалиста |
| `/spec` + `key=value` (одно сообщение) | Заполнить поля профиля пакетом |
| `/profile` | Показать текущий профиль |
| `/settings` | Меню настройки полей по кнопкам |
| **⚙️ Настройки** | То же, что `/settings` |
| **👤 Профиль** | То же, что `/profile` |

### Типичный рабочий процесс

```
1. /start
2. /fio Фамилия Имя Отчество
3. /spec (одним сообщением: position=..., education=..., basis=...)
4. Отправить фото татуировки
5. Получить файл Заключение_специалиста_....docx
```

Для быстрых справочных вопросов без заключения — просто отправьте текст.

---

## Полная карта переменных в `docker/.env`

| Переменная | Где используется | Обязательно менять |
|---|---|---|
| `POSTGRES_PASSWORD` | postgres, migrate, rag, bot | **Да** (для продакшена) |
| `POSTGRES_USER` | postgres, migrate | Нет |
| `POSTGRES_DB` | postgres, migrate | Нет |
| `POSTGRES_DSN` | rag, bot | **Да** (если изменили user/password) |
| `TELEGRAM_BOT_TOKEN` | telegram_bot | **Да** |
| `RAG_URL` | telegram_bot | Нет (внутренний адрес) |
| `LLM_API_URL` | rag | При смене провайдера |
| `LLM_API_KEY` | rag | При использовании OpenAI |
| `LLM_MODEL` | rag | При смене модели |
| `VISION_MODEL` | rag (анализ фото) | При смене VL-модели |
| `PREPARE_VL_API_URL` | prepare_docs | При смене провайдера OCR |
| `PREPARE_VL_MODEL` | prepare_docs | При смене OCR-модели |
| `PREPARE_VL_ZOOM` | prepare_docs | Нет |
| `PREPARE_PAGE_ENGINE` | prepare_docs | Нет (`vl` по умолчанию) |
| `PREPARE_VL_TIMEOUT` | prepare_docs | При медленном VL |
| `EMBEDDING_API_URL` | rag | При смене провайдера эмбеддингов |
| `EMBEDDING_MODEL` | rag | При смене embedding-модели |
| `EMBEDDING_DIM` | rag, БД | **Только с миграцией** (см. Liquibase 006) |
| `EMBEDDING_TIMEOUT_SEC` | rag | При таймаутах на CPU |
| `MULTIMODAL_ENABLED` | rag | `true` только при llama-server |
| `IMAGE_EMBEDDING_*` | rag | При мультимодальности |
| `RAG_PDF_ROOT` | rag | Нет |
| `RAG_DOCS_DIR` | rag | Нет |
| `RAG_TOC_DIR` | rag | Нет |
| `TESSERACT_LANG` | prepare_docs (fallback) | Нет |
| `RAG_TOP_K` | rag | Тонкая настройка |
| `RAG_MAX_DISTANCE` | rag | Тонкая настройка |

---

## Остановка и очистка

```bash
cd docker

# Остановить (данные БД сохраняются)
docker compose down

# Полная очистка (удалить БД и пересоздать)
docker compose down -v
docker compose up -d --build
```

---

## Частые проблемы

| Симптом | Причина | Решение |
|---|---|---|
| «Не удалось обратиться к RAG сервису» (первый запрос) | RAG ещё стартует или индексируется | Подождать 20–30 с, повторить; `docker compose logs rag` |
| `[Индекс: ошибка — Model unloaded]` в конце ответа | Embedding-модель выгружена в LM Studio (часто при смене чат/VL-модели) | Загрузить `EMBEDDING_MODEL`, `docker compose restart rag`; ответ может быть верным — ошибка от фоновой индексации |
| `Errno 101 / Connection refused` для LM Studio | LM Studio не запущен или слушает только 127.0.0.1 | Запустить LM Studio до `compose up`; bind `0.0.0.0` |
| `llama truncation` в логах | Context Length = 4096 | В LM Studio → модель → Context Length ≥ 8192 |
| `timed out` при индексации | Первый инференс CPU > 60 с | Увеличить `EMBEDDING_TIMEOUT_SEC=600` |
| `prepare_docs` падает / VL error | LM Studio недоступен или неверная модель | Проверить `PREPARE_VL_API_URL`, `PREPARE_VL_MODEL` |
| `/health` → `degraded`, `rag_chunk_count=0` | Индексация идёт или упала | `docker compose logs rag` |
| `Token is invalid!` в логах бота | Placeholder вместо токена в `docker/.env` | Токен от @BotFather → `TELEGRAM_BOT_TOKEN=...` → `restart telegram_bot` |
| Бот не присылает `.docx` | Ошибка генерации docx или RAG недоступен | `docker compose logs rag` и `telegram_bot` |
| «Сервис занят» при отправке фото | Идёт обработка предыдущего фото | Дождаться завершения |
| Сменили `EMBEDDING_DIM` | Размерность не совпадает с БД | `docker compose down -v`, перезапуск |
| `TLS handshake timeout` при `--build` | Нет доступа к Docker Hub | `docker compose up -d` без `--build` (если образы уже есть) |
| GitHub: PDF > 50 MB при `git push` | Крупные PDF в `rag/docs/` (~76 MB Балдаев) | Push проходит (лимит 100 MB); опционально `git lfs track "rag/docs/*.pdf"` |
