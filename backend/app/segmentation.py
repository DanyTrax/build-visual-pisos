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
    if len(ys) < 200:
        return -1.0
    coverage = len(ys) / float(mask.size)
    if coverage > 0.78:
        return -1.0
    mean_y = ys.mean() / float(height)
    bottom_ratio = float((ys > height * 0.38).sum()) / len(ys)
    top_ratio = float((ys < height * 0.22).sum()) / len(ys)
    return (mean_y * 1.5) + (bottom_ratio * 2.0) - (top_ratio * 3.0)


def _pick_best_mask(
    masks: list[np.ndarray],
    img_shape: tuple[int, int, int],
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    h = img_shape[0]
    best_score = -1.0
    best_mask = None
    for mask in masks:
        mask = _resize_mask(mask, img_shape[:2])
        if exclude_mask is not None and (exclude_mask > 127).any():
            mask = _subtract_masks(mask, exclude_mask, dilate_iters=2, kernel_size=11)
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


def _subtract_masks(
    base_mask: np.ndarray,
    remove_mask: np.ndarray,
    dilate_iters: int = 2,
    kernel_size: int = 11,
) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded = cv2.dilate(remove_mask, kernel, iterations=dilate_iters)
    cleaned = base_mask.copy()
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


def _detect_environment(
    client: Any,
    tmp_path: str,
    config: dict,
    shape: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    env_prompt = config.get(
        "environment_prompt",
        (
            "grass . lawn . sky . clouds . water . lake . hill . tree . bush . plant . flower . "
            "fence . wall . building . window . person . dog . car . umbrella . pot . railing . stairs"
        ),
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

        exclude_mask = environment_mask

        floor_prompt = config.get(
            "floor_text_prompt",
            "floor . patio . stone floor . tile floor . wooden floor . ground",
        )
        negative_prompt = config.get(
            "negative_mask_prompt",
            "sofa . couch . chair . table . furniture . bed . desk . person . wall . grass . bush",
        )
        floor_adj = int(config.get("mask_adjustment_factor", 6))

        floor_masks = _run_grounded_sam(client, tmp_path, config, floor_prompt, negative_prompt, floor_adj)
        if not floor_masks:
            floor = heuristic_floor_mask(img_bgr)
            return SegmentationResult(
                floor_mask=floor,
                environment_mask=environment_mask,
                raw_floor_mask=floor,
                message="fallback: Replicate no devolvio mascaras de piso",
            )

        raw_floor = _pick_best_mask(floor_masks, img_bgr.shape, exclude_mask=exclude_mask)
        if raw_floor is None:
            floor = heuristic_floor_mask(img_bgr)
            return SegmentationResult(
                floor_mask=floor,
                environment_mask=environment_mask,
                raw_floor_mask=floor,
                message="fallback: ninguna mascara valida para piso",
            )

        floor = raw_floor.copy()
        if (furniture_mask > 127).any():
            floor = _subtract_masks(floor, furniture_mask, dilate_iters=5, kernel_size=17)
            steps.append("restado muebles")
        elif (environment_mask > 127).any():
            floor = _subtract_masks(floor, environment_mask, dilate_iters=3, kernel_size=13)
            steps.append("restado entorno")

        floor = refine_floor_remove_foreground_objects(img_bgr, floor)
        if "muebles" in steps or "entorno" in steps:
            steps.append("filtro color")

        processed = postprocess_floor_mask(floor, shape)
        if float((processed > 30).mean()) < 0.03:
            floor_h = heuristic_floor_mask(img_bgr)
            return SegmentationResult(
                floor_mask=floor_h,
                environment_mask=environment_mask,
                raw_floor_mask=raw_floor,
                message="fallback: mascara IA demasiado pequena tras excluir muebles",
            )

        msg = "ok: " + " + ".join(steps) if steps else "ok: segmentacion IA (Grounded SAM)"
        return SegmentationResult(
            floor_mask=processed,
            environment_mask=environment_mask,
            raw_floor_mask=postprocess_floor_mask(raw_floor, shape),
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
