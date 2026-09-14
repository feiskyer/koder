"""Helpers for attaching local images to a multimodal user turn.

The CLI exposes a repeatable ``-i/--image`` flag (see
:func:`koder_agent.cli._build_cli_parser`).  Each path is validated, read, and
base64-encoded into a ``data:`` URL, then wrapped in an ``input_image`` content
block.  These blocks are prepended to the first user turn so the initial
message becomes multimodal (image(s) + text).

The block shape follows the OpenAI Responses input schema that the
``openai-agents`` SDK / LiteLLM expect:

``{"type": "input_image", "image_url": "data:image/png;base64,...", "detail": "auto"}``

and text is carried as ``{"type": "input_text", "text": "..."}``.  A user turn
with mixed content is a single message dict::

    {"role": "user", "content": [<image blocks...>, <text block>]}

which ``Runner.run`` accepts as a ``list[TResponseInputItem]`` input.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

__all__ = [
    "ImageInputError",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "model_supports_vision",
    "detect_image_mime_type",
    "encode_image_data_url",
    "build_image_content_block",
    "build_image_content_blocks",
    "build_multimodal_input",
]


class ImageInputError(ValueError):
    """Raised when an ``--image`` path is missing or not a readable image."""


# Extension -> MIME type. These are the raster formats broadly accepted by
# vision-capable providers (OpenAI, Anthropic, Gemini via LiteLLM).
SUPPORTED_IMAGE_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

# Leading magic bytes for the formats above. Used as a second line of defence
# so a mislabeled ``.png`` that is actually text is rejected up front.
_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


def model_supports_vision(model: str) -> bool:
    """Return whether ``model`` is known to accept image input.

    Uses LiteLLM's capability registry when available. Unknown models return
    ``False`` here; callers may still choose to attach and let the provider
    decide (see the CLI, which warns rather than refuses).
    """
    if not model:
        return False
    try:
        from ..litellm_cost_map import get_litellm
        from .model_info import get_model_name_variants_for_lookup

        litellm = get_litellm()
        for name in get_model_name_variants_for_lookup(model):
            try:
                if litellm.supports_vision(model=name):
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def _detect_mime_from_magic(head: bytes) -> Optional[str]:
    for signature, mime in _MAGIC_SIGNATURES:
        if head.startswith(signature):
            return mime
    # WEBP: "RIFF" .... "WEBP"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


def detect_image_mime_type(path: Path, data: bytes) -> str:
    """Determine the MIME type for ``path`` from magic bytes, then extension.

    Raises:
        ImageInputError: if the file does not look like a supported image.
    """
    magic_mime = _detect_mime_from_magic(data[:16])
    if magic_mime is not None:
        return magic_mime

    ext_mime = SUPPORTED_IMAGE_EXTENSIONS.get(path.suffix.lower())
    if ext_mime is not None:
        # Extension claims a supported format but the bytes did not match any
        # known signature — treat as not an image to avoid sending garbage.
        raise ImageInputError(
            f"File does not appear to be a valid image: {path} "
            f"(extension is {path.suffix} but content is unrecognized)."
        )

    raise ImageInputError(
        f"Unsupported image type: {path}. Supported extensions: "
        f"{', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}."
    )


def _read_image_bytes(image_path: str) -> tuple[Path, bytes]:
    path = Path(image_path).expanduser()
    if not path.exists():
        raise ImageInputError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise ImageInputError(f"Image path is not a file: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ImageInputError(f"Could not read image file: {path} ({exc}).") from exc
    if not data:
        raise ImageInputError(f"Image file is empty: {path}")
    return path, data


def encode_image_data_url(image_path: str) -> str:
    """Read ``image_path`` and return a ``data:<mime>;base64,<data>`` URL.

    Raises:
        ImageInputError: if the path is missing or not a valid image.
    """
    path, data = _read_image_bytes(image_path)
    mime = detect_image_mime_type(path, data)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_image_content_block(image_path: str, *, detail: str = "auto") -> dict:
    """Build a single ``input_image`` content block for ``image_path``.

    Raises:
        ImageInputError: if the path is missing or not a valid image.
    """
    return {
        "type": "input_image",
        "image_url": encode_image_data_url(image_path),
        "detail": detail,
    }


def build_image_content_blocks(image_paths, *, detail: str = "auto") -> list[dict]:
    """Build ``input_image`` blocks for every path in ``image_paths``.

    Each path is validated eagerly; the first bad path raises so the user gets
    a clear error before any turn is dispatched.
    """
    blocks: list[dict] = []
    for image_path in image_paths or []:
        blocks.append(build_image_content_block(image_path, detail=detail))
    return blocks


def build_multimodal_input(text: Optional[str], image_paths, *, detail: str = "auto"):
    """Construct the initial user turn, multimodal when images are present.

    Returns:
        - ``None`` when there is neither text nor any image (nothing to send).
        - the plain ``text`` string when there are no images (unchanged path).
        - a ``list`` with one user-message dict whose ``content`` is a list of
          image blocks followed by a text block, suitable as ``Runner.run``
          input, when images are present.

    Raises:
        ImageInputError: if any image path is missing or not a valid image.
    """
    image_blocks = build_image_content_blocks(image_paths, detail=detail)
    if not image_blocks:
        return text

    content: list[dict] = list(image_blocks)
    if text:
        content.append({"type": "input_text", "text": text})

    return [{"role": "user", "content": content}]
