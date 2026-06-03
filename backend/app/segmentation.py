import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .image_ops import (
    heuristic_floor_mask,
    postprocess_floor_mask,
    refine_floor_remove_foreground_objects,
)


@dataclass
class SegmentationResult:
    floor_mask: np.ndarray
    environment_mask: np.ndarray
    raw_floor_mask: np.ndarray
    message: str


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
    if len(ys) < 150:
        return -1.0
    coverage = len(ys) / float(mask.size)
    if coverage > 0.88:
        return -1.0
    mean_y = ys.mean() / float(height)
    bottom_ratio = float((ys > height * 0.35).sum()) / len(ys)
    top_ratio = float((ys < height * 0.15).sum()) / len(ys)
    coverage_bonus = min(coverage, 0.55) * 0.8
    return (mean_y * 1.2) + (bottom_ratio * 1.5) + coverage_bonus - (top_ratio * 2.5)


def _merge_floor_masks(masks: list[np.ndarray], img_shape: tuple[int, int, int]) -> np.ndarray | None:
    """Une todas las mascaras de piso validas (evita huecos entre baldosas)."""
    h, w = img_shape[:2]
    merged = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        m = _resize_mask(mask, (h, w))
        if _score_floor_mask(m, h) < 0:
            continue
        merged = np.maximum(merged, m)
    if int((merged > 127).sum()) < 150:
        return None
    return merged


def _subtract_overlap_only(
    base_mask: np.ndarray,
    remove_mask: np.ndarray,
    dilate_iters: int = 3,
    kernel_size: int = 11,
) -> np.ndarray:
    overlap = cv2.bitwise_and(base_mask, remove_mask)
    if int((overlap > 127).sum()) < 40:
        return base_mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded = cv2.dilate(overlap, kernel, iterations=dilate_iters)
    cleaned = base_mask.copy()
    cleaned[expanded > 127] = 0
    return cleaned


def _union_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    union = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        mask = _resize_mask(mask, shape)
        union = np.maximum(union, mask)
    return union


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


def _detect_environment(
    client: Any,
    tmp_path: str,
    config: dict,
    shape: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    env_prompt = config.get(
        "environment_prompt",
        "grass . lawn . sky . clouds . water . lake . hill . tree . bush . plant . flower . person . umbrella . pot . planter . railing",
    )
    env_masks = _run_grounded_sam(client, tmp_path, config, env_prompt, "", 0)
    if not env_masks:
        return np.zeros(shape, dtype=np.uint8), False
    return _union_masks(env_masks, shape), True


def _detect_furniture(
    client: Any,
    tmp_path: str,
    config: dict,
    shape: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    furn_prompt = config.get(
        "furniture_subtraction_prompt",
        (
            "sofa . couch . loveseat . armchair . chair . seat . cushion . pillow . "
            "coffee table . table . outdoor furniture . patio furniture . ottoman . furniture"
        ),
    )
    furn_masks = _run_grounded_sam(client, tmp_path, config, furn_prompt, "", 0)
    if not furn_masks:
        return np.zeros(shape, dtype=np.uint8), False
    return _union_masks(furn_masks, shape), True


def segment_floor_with_replicate(img_bytes: bytes, img_bgr: np.ndarray, config: dict, token: str) -> SegmentationResult:
    empty_env = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    if not token:
        floor = heuristic_floor_mask(img_bgr)
        return SegmentationResult(
            floor_mask=floor,
            environment_mask=empty_env,
            raw_floor_mask=floor,
            message="fallback: falta REPLICATE_API_TOKEN en .env",
        )

    tmp_path = None
    try:
        import replicate

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp.flush()
            tmp_path = tmp.name

        client = replicate.Client(api_token=token)
        shape = img_bgr.shape[:2]
        steps: list[str] = []
        top_crop = float(config.get("floor_top_crop_ratio", 0.08))

        environment_mask = empty_env
        furniture_mask = empty_env

        if config.get("enable_environment_layer", True):
            environment_mask, env_ok = _detect_environment(client, tmp_path, config, shape)
            if env_ok:
                steps.append("entorno")

        if config.get("enable_furniture_subtraction", True):
            furniture_mask, furn_ok = _detect_furniture(client, tmp_path, config, shape)
            if furn_ok:
                steps.append("muebles")
            environment_mask = np.maximum(environment_mask, furniture_mask)

        floor_prompt = config.get(
            "floor_text_prompt",
            "floor . patio . stone floor . tile floor . ceramic tile . wooden floor . ground",
        )
        floor_prompt_alt = (config.get("floor_text_prompt_alt") or "").strip()
        negative_prompt = config.get(
            "negative_mask_prompt",
            "sofa . couch . chair . table . furniture . bed . desk . person . grass . bush",
        )
        floor_adj = int(config.get("mask_adjustment_factor", 8))

        floor_masks = _run_grounded_sam(client, tmp_path, config, floor_prompt, negative_prompt, floor_adj)
        if floor_prompt_alt:
            floor_masks += _run_grounded_sam(client, tmp_path, config, floor_prompt_alt, negative_prompt, floor_adj)

        if not floor_masks:
            floor = heuristic_floor_mask(img_bgr)
            return SegmentationResult(
                floor_mask=floor,
                environment_mask=environment_mask,
                raw_floor_mask=floor,
                message="fallback: Replicate no devolvio mascaras de piso",
            )

        raw_floor = _merge_floor_masks(floor_masks, img_bgr.shape)
        if raw_floor is None:
            floor = heuristic_floor_mask(img_bgr)
            return SegmentationResult(
                floor_mask=floor,
                environment_mask=environment_mask,
                raw_floor_mask=floor,
                message="fallback: ninguna mascara valida para piso",
            )
        steps.append("union piso")

        floor = raw_floor.copy()
        if (furniture_mask > 127).any():
            floor = _subtract_overlap_only(floor, furniture_mask, dilate_iters=4, kernel_size=15)
            steps.append("restado muebles")
        if (environment_mask > 127).any():
            env_only = cv2.subtract(environment_mask, furniture_mask)
            floor = _subtract_overlap_only(floor, env_only, dilate_iters=3, kernel_size=11)
            steps.append("restado entorno")

        if config.get("enable_color_refinement", False):
            floor = refine_floor_remove_foreground_objects(img_bgr, floor)
            steps.append("filtro color")

        processed = postprocess_floor_mask(floor, shape, top_crop_ratio=top_crop)
        if float((processed > 30).mean()) < 0.03:
            floor_h = heuristic_floor_mask(img_bgr)
            return SegmentationResult(
                floor_mask=floor_h,
                environment_mask=environment_mask,
                raw_floor_mask=raw_floor,
                message="fallback: mascara demasiado pequena",
            )

        msg = "ok: " + " + ".join(steps) if steps else "ok: segmentacion IA"
        return SegmentationResult(
            floor_mask=processed,
            environment_mask=environment_mask,
            raw_floor_mask=postprocess_floor_mask(raw_floor, shape, top_crop_ratio=top_crop),
            message=msg,
        )
    except Exception as exc:  # noqa: BLE001
        floor = heuristic_floor_mask(img_bgr)
        return SegmentationResult(
            floor_mask=floor,
            environment_mask=empty_env,
            raw_floor_mask=floor,
            message=f"fallback: error Replicate ({exc})",
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
