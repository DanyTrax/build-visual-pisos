import base64
from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image_from_bytes(raw: bytes, max_width: int = 1280) -> np.ndarray:
    pil = Image.open(BytesIO(raw))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    if pil.width > max_width:
        ratio = max_width / float(pil.width)
        pil = pil.resize((max_width, int(pil.height * ratio)))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def encode_image_base64(img_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("No se pudo codificar la imagen")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def create_overlay(original_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    color = np.zeros_like(original_bgr)
    color[:, :, 1] = 180
    alpha = (mask / 255.0 * 0.45)[:, :, None]
    merged = (original_bgr.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
    return merged


def heuristic_floor_mask(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    y_start = int(h * 0.48)
    mask[y_start:h, :] = 255
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    return mask


def make_tiled_texture(texture_bgr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    tex_h, tex_w = texture_bgr.shape[:2]
    rep_y = max(1, (h // tex_h) + 2)
    rep_x = max(1, (w // tex_w) + 2)
    tiled = np.tile(texture_bgr, (rep_y, rep_x, 1))
    return tiled[:h, :w]


def perspective_texture(texture_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 20)
    if len(xs) < 50:
        return texture_bgr
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    src = np.float32([[0, h], [w, h], [0, 0], [w, 0]])
    shrink = int((x_max - x_min) * 0.25)
    top_left = max(0, x_min + shrink)
    top_right = min(w - 1, x_max - shrink)
    dst = np.float32(
        [
            [x_min, y_max],
            [x_max, y_max],
            [top_left, y_min],
            [top_right, y_min],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(texture_bgr, matrix, (w, h), flags=cv2.INTER_LINEAR)


def blend_floor(original_bgr: np.ndarray, floor_texture_bgr: np.ndarray, mask: np.ndarray, feather_px: int, strength: float) -> np.ndarray:
    feather_px = max(3, feather_px | 1)
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (feather_px, feather_px), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]

    base = original_bgr.astype(np.float32) / 255.0
    tex = floor_texture_bgr.astype(np.float32) / 255.0

    multiplied = np.clip(base * tex, 0.0, 1.0)
    mixed = (1.0 - strength) * tex + strength * multiplied
    out = base * (1.0 - alpha) + mixed * alpha
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
