"""
Visualization helpers - draw detection boxes on images.
"""

from __future__ import annotations
from typing import Iterable
from PIL import Image, ImageDraw, ImageFont
from models.schemas import Detection

# A simple, colorblind-friendly palette
_BOX_COLORS: dict[str, str] = {
    "person":  "#E63946",   # red
    "knife":   "#1D3557",   # dark blue
    "weapon":  "#1D3557",
    "vehicle": "#2A9D8F",   # teal
    "bag":     "#F4A261",   # orange
    "bottle":  "#8338EC",   # purple
}

def _color_for(label: str) -> str:
    return _BOX_COLORS.get(label, "#264653")

def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()

def draw_detections(
    image: Image.Image,
    detections: Iterable[Detection],
    box_width: int | None = None,
) -> Image.Image:
    """Return a copy of `image` with each detection drawn on it."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    img_w, img_h = image.size
    
    # Dynamic box width: 0.5% of image width, minimum 3px
    if box_width is None:
        box_width = max(3, int(img_w * 0.005))
    
    # Dynamic font size: 1.5% of image height, minimum 14px
    font_size = max(14, int(img_h * 0.015))
    font = _font(font_size)

    for det in detections:
        x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
        color = _color_for(det.label)

        # 1. Draw the Bounding Box
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=box_width)

        # 2. Draw Label + Confidence
        text = f"{det.label} {det.confidence:.2f}"
        
        try:
            text_w = draw.textlength(text, font=font)
        except AttributeError:
            text_w = len(text) * (font_size // 2)
            
        text_bg_h = font_size + 4

        # Label Background
        draw.rectangle(
            [(x1, max(0, y1 - text_bg_h)), (x1 + text_w + 8, y1)],
            fill=color,
        )
        # Label Text
        draw.text((x1 + 4, max(0, y1 - text_bg_h + 2)), text, fill="white", font=font)

    return annotated
