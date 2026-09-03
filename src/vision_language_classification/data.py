from __future__ import annotations

from pathlib import Path

from .schema import Example, stable_id


def _source_video_id(row: dict) -> str:
    explicit = row.get("source_video_id") or row.get("video_id")
    if explicit:
        return str(explicit)
    source = row.get("source_path") or row.get("path") or row.get("image_path")
    return Path(str(source)).stem


def from_video_curation(row: dict, frame_path: str | None = None) -> Example:
    """Convert a Video-Curation enriched-manifest row into the common schema.

    `frame_path` should identify the middle frame captioned by BLIP-2. If absent,
    an existing image_path is required; video decoding is deliberately separate.
    """
    image_path = frame_path or row.get("image_path")
    if not image_path:
        raise ValueError("row needs image_path or an explicitly extracted frame_path")
    video_id = _source_video_id(row)
    caption = row.get("caption")
    if not caption:
        raise ValueError("row has no caption; fallback captions are not accepted as BLIP-2 labels")
    return Example(
        example_id=stable_id(video_id, str(image_path)),
        image_path=str(image_path),
        caption=str(caption),
        source_video_id=video_id,
        source_category=row.get("label") or row.get("source_category"),
        blur_score=_optional_float(row.get("blur_score")),
        texture_score=_optional_float(row.get("texture_score")),
        split=row.get("split"),
    )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
