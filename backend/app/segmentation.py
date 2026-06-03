import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from PIL import Image

from .image_ops import heuristic_floor_mask, postprocess_floor_mask


def _mask_from_bytes(raw: bytes) -> np.ndarray:
    img = Image.open(BytesIO(raw)).convert("L")
    return np.array(img, dtype=np.uint8)


def _download_mask(source: Any) -> np.ndarray:
    if hasattr(source, "read"):
        return _mask_from_bytes(source.read())
    if isinstance(source, str):
        with httpx.Client(timeout=90.0) as client:
            response = client.get(source)
            response.raise_for_status()
        return _mask_from_bytes(response.content)
    raise ValueError("Formato de mascara no soportado")


def _collect_outputs(output: Any) -> list[Any]:
    if output is None:
        return []
    if isinstance(output, list):
        return output
    return [output]


def _score_floor_mask(mask: np.ndarray, height: int) -> float:
    ys, _ = np.where(mask > 127)
    if len(ys) < 200:
        return -1.0
    coverage = len(ys) / float(mask.size)
    if coverage > 0.82:
        return -1.0
    mean_y = ys.mean() / float(height)
    bottom_ratio = float((ys > height * 0.38).sum()) / len(ys)
    top_ratio = float((ys < height * 0.22).sum()) / len(ys)
    return (mean_y * 1.5) + (bottom_ratio * 2.0) - (top_ratio * 3.0)


def _pick_best_mask(masks: list[np.ndarray], img_shape: tuple[int, int, int]) -> np.ndarray | None:
    h = img_shape[0]
    best_score = -1.0
    best_mask = None
    for mask in masks:
        if mask.shape[:2] != (h, img_shape[1]):
            mask = cv2.resize(mask, (img_shape[1], h), interpolation=cv2.INTER_NEAREST)
        score = _score_floor_mask(mask, h)
        if score > best_score:
            best_score = score
            best_mask = mask
    return best_mask


def segment_floor_with_replicate(img_bytes: bytes, img_bgr: np.ndarray, config: dict, token: str) -> tuple[np.ndarray, str]:
    if not token:
        return heuristic_floor_mask(img_bgr), "fallback: falta REPLICATE_API_TOKEN en .env"

    tmp_path = None
    try:
        import replicate

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp.flush()
            tmp_path = tmp.name

        client = replicate.Client(api_token=token)
        with open(tmp_path, "rb") as image_file:
            output = client.run(
                config["replicate_model"],
                input={
                    "image": image_file,
                    "mask_prompt": config.get("floor_text_prompt", "floor . ground . flooring"),
                    "negative_mask_prompt": config.get("negative_mask_prompt", ""),
                    "adjustment_factor": int(config.get("mask_adjustment_factor", 12)),
                },
            )

        masks: list[np.ndarray] = []
        for item in _collect_outputs(output):
            try:
                masks.append(_download_mask(item))
            except Exception:
                continue

        if not masks:
            return heuristic_floor_mask(img_bgr), "fallback: Replicate no devolvio mascaras"

        best = _pick_best_mask(masks, img_bgr.shape)
        if best is None:
            return heuristic_floor_mask(img_bgr), "fallback: ninguna mascara valida para piso"

        processed = postprocess_floor_mask(best, img_bgr.shape[:2])
        if float((processed > 30).mean()) < 0.03:
            return heuristic_floor_mask(img_bgr), "fallback: mascara IA demasiado pequena"

        return processed, "ok: segmentacion IA (Grounded SAM)"
    except Exception as exc:  # noqa: BLE001
        return heuristic_floor_mask(img_bgr), f"fallback: error Replicate ({exc})"
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
