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
    """Respaldo aproximado: prioriza zona inferior visible (no cubre muebles en zona alta)."""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 40], dtype=np.uint8)
    upper = np.array([180, 80, 220], dtype=np.uint8)
    color_mask = cv2.inRange(hsv, lower, upper)

    y_grid = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    weight = np.clip((y_grid - 0.25) / 0.75, 0, 1)
    weighted = (color_mask.astype(np.float32) * weight).astype(np.uint8)
    _, mask = cv2.threshold(weighted, 40, 255, cv2.THRESH_BINARY)
    return postprocess_floor_mask(mask, (h, w))


def postprocess_floor_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    mask[: int(h * 0.18), :] = 0
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        bottom_band = int(h * 0.92)
        kept = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            touches_bottom = (y + ch) >= bottom_band
            area = cv2.contourArea(contour)
            if touches_bottom and area > (h * w * 0.005):
                kept.append(contour)
        if not kept:
            kept = [max(contours, key=cv2.contourArea)]
        cleaned = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cleaned, kept, -1, 255, thickness=-1)
        mask = cleaned

    mask = cv2.GaussianBlur(mask, (9, 9), 0)
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
