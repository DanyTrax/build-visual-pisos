from io import BytesIO

import httpx
import numpy as np
from PIL import Image

from .image_ops import heuristic_floor_mask


def _download_mask(mask_url: str) -> np.ndarray:
    with httpx.Client(timeout=60.0) as client:
        response = client.get(mask_url)
        response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("L")
    return np.array(img, dtype=np.uint8)


def segment_floor_with_replicate(img_bytes: bytes, img_bgr: np.ndarray, config: dict, token: str) -> tuple[np.ndarray, str]:
    """
    Intenta segmentar piso con Replicate Grounded SAM.
    Si falla o no hay token, usa mascara heuristica para entorno local/pruebas.
    """
    if not token:
        return heuristic_floor_mask(img_bgr), "fallback: token replicado no configurado"

    try:
        import replicate  # lazy import

        client = replicate.Client(api_token=token)
        output = client.run(
            config["replicate_model"],
            input={
                "image": BytesIO(img_bytes),
                "mask_prompt": config["floor_text_prompt"],
                "negative_mask_prompt": config.get("negative_mask_prompt", ""),
            },
        )
        mask_url = None
        if isinstance(output, list) and output:
            mask_url = output[0]
        elif isinstance(output, str):
            mask_url = output
        if not mask_url:
            return heuristic_floor_mask(img_bgr), "fallback: respuesta replicada vacia"
        mask = _download_mask(mask_url)
        if mask.shape != img_bgr.shape[:2]:
            from cv2 import resize, INTER_NEAREST

            mask = resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=INTER_NEAREST)
        if float((mask > 30).mean()) < 0.02:
            return heuristic_floor_mask(img_bgr), "fallback: mascara muy pequena"
        return mask, "ok"
    except Exception as exc:  # noqa: BLE001
        return heuristic_floor_mask(img_bgr), f"fallback: error replicate ({exc})"
