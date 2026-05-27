"""
Small Korean text helpers for OpenCV frames.

OpenCV's built-in Hershey font cannot render Hangul.  These helpers use PIL and
the Windows Malgun Gothic font when available, then copy the result back into
the BGR OpenCV image.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - fallback for minimal environments.
    Image = None
    ImageDraw = None
    ImageFont = None


FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("C:/Windows/Fonts/malgunbd.ttf"),
    Path("C:/Windows/Fonts/gulim.ttc"),
]


@lru_cache(maxsize=16)
def _load_font(size: int):
    if ImageFont is None:
        return None
    for path in FONT_CANDIDATES:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_korean_text(
    vis: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_size: int = 15,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw Korean text on a BGR OpenCV frame."""
    if Image is None or ImageDraw is None:
        cv2.putText(vis, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return

    font = _load_font(font_size)
    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    b, g, r = color
    draw.text(org, str(text), font=font, fill=(r, g, b))
    vis[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_korean_lines(
    vis: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    org: tuple[int, int],
    font_size: int = 15,
    line_gap: int = 20,
) -> None:
    x, y = org
    for idx, (text, color) in enumerate(lines):
        draw_korean_text(vis, text, (x, y + idx * line_gap), font_size=font_size, color=color)
