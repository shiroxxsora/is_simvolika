"""MIME и даунскейл изображений перед отправкой в RAG."""

from __future__ import annotations

import io
from typing import Final

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[misc, assignment]
    ImageOps = None  # type: ignore[misc, assignment]

_DEFAULT_MAX_SIDE: Final[int] = 2048


def guess_mime_for_telegram_photo() -> str:
    """Сжатые размеры фото в Telegram — обычно JPEG."""
    return "image/jpeg"


def guess_mime_for_document(mime_type: str | None) -> str | None:
    if not mime_type:
        return None
    mt = mime_type.lower().strip()
    if mt.startswith("image/"):
        return mt
    return None


def maybe_downscale_image(
    image_bytes: bytes,
    *,
    max_side: int = _DEFAULT_MAX_SIDE,
    mime_type: str = "image/jpeg",
) -> tuple[bytes, str]:
    """Уменьшает длинную сторону до max_side; при ошибке или без PIL возвращает исходные байты."""
    if Image is None:
        return image_bytes, mime_type
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            if max(w, h) <= max_side:
                return image_bytes, mime_type
            scale = max_side / float(max(w, h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
            im = im.convert("RGB")
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=90)
            return out.getvalue(), "image/jpeg"
    except (OSError, ValueError, TypeError):
        return image_bytes, mime_type
