import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .image_ops import heuristic_floor_mask, postprocess_floor_mask


def _mask_from_bytes(raw: bytes) -> np.ndarray:
    img = Image.open(BytesIO(raw)).convert("L")
    return np.array(img, dtype=np.uint8)


def _download_mask(source: Any) -> np.ndarray:
    if hasattr(source, "read"):
        return _mask_from_bytes(source.read())
    url = str(source)
    import httpx

    with httpx.Client(timeout=90.0) as client:
        response = client.get(url)
        response.raise_for_status()
    return _mask_from_bytes(response.content)


def _collect_outputs(output: Any) -> list[Any]:
    if output is None:
        return []
    if isinstance(output, list):
        return output
    return [output]


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask.shape[:2] != (h, w):
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def _score_floor_mask(mask: np.ndarray, height: int) -> float:
    ys, _ = np.where(mask > 127)
    if len(ys) < 200:
        return -1.0
    coverage = len(ys) / float(mask.size)
    if coverage > 0.78:
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
        mask = _resize_mask(mask, img_shape[:2])
        score = _score_floor_mask(mask, h)
        if score > best_score:
            best_score = score
            best_mask = mask
    return best_mask


def _union_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    union = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        mask = _resize_mask(mask, shape)
        union = np.maximum(union, mask)
    return union


def _subtract_object_mask(floor_mask: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((9, 9), np.uint8)
    expanded = cv2.dilate(object_mask, kernel, iterations=2)
    cleaned = floor_mask.copy()
    cleaned[expanded > 127] = 0
    return cleaned


def _run_grounded_sam(
    client: Any,
    tmp_path: str,
    config: dict,
    mask_prompt: str,
    negative_mask_prompt: str,
    adjustment_factor: int,
) -> list[np.ndarray]:
    with open(tmp_path, "rb") as image_file:
        output = client.run(
            config["replicate_model"],
            input={
                "image": image_file,
                "mask_prompt": mask_prompt,
                "negative_mask_prompt": negative_mask_prompt,
                "adjustment_factor": adjustment_factor,
            },
        )

    masks: list[np.ndarray] = []
    for item in _collect_outputs(output):
        try:
            masks.append(_download_mask(item))
        except Exception:
            continue
    return masks


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
        floor_prompt = config.get(
            "floor_text_prompt",
            "floor . patio . stone floor . tile floor . wooden floor . ground",
        )
        negative_prompt = config.get(
            "negative_mask_prompt",
            "sofa . couch . chair . table . furniture . bed . desk . person . wall . plant",
        )
        floor_adj = int(config.get("mask_adjustment_factor", 6))

        floor_masks = _run_grounded_sam(client, tmp_path, config, floor_prompt, negative_prompt, floor_adj)
        if not floor_masks:
            return heuristic_floor_mask(img_bgr), "fallback: Replicate no devolvio mascaras de piso"

        best = _pick_best_mask(floor_masks, img_bgr.shape)
        if best is None:
            return heuristic_floor_mask(img_bgr), "fallback: ninguna mascara valida para piso"

        subtracted = False
        if config.get("enable_object_subtraction", True):
            obj_prompt = config.get(
                "objects_subtraction_prompt",
                "sofa . couch . coffee table . chair . table . outdoor furniture . ottoman . cushion . furniture",
            )
            obj_masks = _run_grounded_sam(client, tmp_path, config, obj_prompt, "", 0)
            if obj_masks:
                obj_union = _union_masks(obj_masks, img_bgr.shape[:2])
                best = _subtract_object_mask(best, obj_union)
                subtracted = True

        processed = postprocess_floor_mask(best, img_bgr.shape[:2])
        if float((processed > 30).mean()) < 0.03:
            return heuristic_floor_mask(img_bgr), "fallback: mascara IA demasiado pequena"

        msg = "ok: segmentacion IA (Grounded SAM)"
        if subtracted:
            msg += " + exclusion de muebles"
        return processed, msg
    except Exception as exc:  # noqa: BLE001
        return heuristic_floor_mask(img_bgr), f"fallback: error Replicate ({exc})"
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
