"""Pre-labeling service using OpenAI vision API for real object detection."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from uuid import UUID

from openai import AsyncOpenAI

from config import settings
from models import AnnotationType, Asset, LabelClass

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert autonomous-vehicle perception annotator.
Given an image, detect all visible objects that match the requested label classes.
For each detection return a bounding box in pixel coordinates relative to the
image dimensions provided.

Rules:
- Only return objects you are confident about (confidence >= 0.5).
- Each bounding box is {x, y, width, height} where (x, y) is the top-left corner.
- Coordinates must be integers within the image bounds.
- Return a JSON object with a single key "detections" containing an array.
- Each detection has: "label" (string, must match one of the class names exactly),
  "confidence" (float 0-1), "x" (int), "y" (int), "width" (int), "height" (int).
- If no objects are found, return {"detections": []}.
- Do NOT include any text outside the JSON object.
"""


def _build_user_prompt(label_classes: list[LabelClass], img_w: int, img_h: int) -> str:
    class_names = [lc.name for lc in label_classes]
    return (
        f"Image dimensions: {img_w}x{img_h} pixels.\n"
        f"Detect all objects belonging to these classes: {', '.join(class_names)}.\n"
        f"Return the JSON detections array."
    )


def _image_to_data_url(image_path: Path, mime_type: str = "image/jpeg") -> str:
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:{mime_type};base64,{b64}"


async def generate_openai_predictions(
    asset: Asset,
    label_classes: list[LabelClass],
    *,
    model: str | None = None,
) -> list[dict]:
    """Call OpenAI vision model to detect objects and return prediction dicts."""
    if not settings.openai_api_key or settings.openai_api_key == "your-openai-api-key-here":
        logger.warning("OPENAI_API_KEY not configured, falling back to empty predictions")
        return []

    img_w = asset.width or 800
    img_h = asset.height or 600

    image_path = settings.local_storage_dir / asset.storage_key
    if not image_path.exists():
        logger.warning("Image file not found: %s", image_path)
        return []

    data_url = _image_to_data_url(image_path, asset.mime_type or "image/jpeg")

    class_name_to_lc: dict[str, LabelClass] = {lc.name.lower(): lc for lc in label_classes}

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    model_name = model or settings.openai_model

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _build_user_prompt(label_classes, img_w, img_h),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=2048,
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        detections = data.get("detections", [])

    except Exception:
        logger.exception("OpenAI vision API call failed for asset %s", asset.id)
        return []

    predictions: list[dict] = []
    for det in detections:
        label = det.get("label", "").lower()
        lc = class_name_to_lc.get(label)
        if not lc:
            continue

        confidence = float(det.get("confidence", 0.0))
        if confidence < 0.3:
            continue

        x = max(0, int(det.get("x", 0)))
        y = max(0, int(det.get("y", 0)))
        w = max(1, int(det.get("width", 1)))
        h = max(1, int(det.get("height", 1)))

        x = min(x, img_w - 1)
        y = min(y, img_h - 1)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        predictions.append(
            {
                "label_class_id": lc.id,
                "type": AnnotationType.bbox,
                "geometry": {"x": x, "y": y, "width": w, "height": h},
                "confidence": round(confidence, 2),
            }
        )

    logger.info(
        "OpenAI detected %d objects in asset %s (%s)",
        len(predictions), asset.id, asset.filename,
    )
    return predictions
