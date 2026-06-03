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


def create_overlay(
    original_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int] = (0, 180, 0),
    alpha_scale: float = 0.45,
) -> np.ndarray:
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    color = np.zeros_like(original_bgr)
    color[:, :, 0] = color_bgr[0]
    color[:, :, 1] = color_bgr[1]
    color[:, :, 2] = color_bgr[2]
    alpha = (mask / 255.0 * alpha_scale)[:, :, None]
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


def fill_floor_mask_continuous(mask: np.ndarray) -> np.ndarray:
    """Cierra huecos entre baldosas y reconecta el piso hacia el fondo de la escena."""
    h, w = mask.shape[:2]
    m = np.clip(mask, 0, 255).astype(np.uint8)
    k = max(17, int(min(h, w) * 0.025) | 1)
    kernel = np.ones((k, k), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)

    filled = m.copy()
    flood = np.zeros((h + 2, w + 2), np.uint8)
    step = max(1, w // 16)
    for x in range(0, w, step):
        for y in (h - 1, h - 2, h - 3):
            if y < 0 or filled[y, x] <= 127:
                continue
            cv2.floodFill(filled, flood, (int(x), int(y)), 255)

    if int((filled > 127).sum()) < int((m > 127).sum()) * 0.55:
        filled = m

    for y in range(h - 1, int(h * 0.12), -1):
        row = filled[y, :]
        if (row > 127).mean() < 0.2:
            continue
        gap = cv2.morphologyEx(row.reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, 31), np.uint8))
        filled[y, :] = np.maximum(row, gap.reshape(-1))

    return filled


def postprocess_floor_mask(mask: np.ndarray, shape: tuple[int, int], top_crop_ratio: float = 0.08) -> np.ndarray:
    h, w = shape
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    mask[: max(1, int(h * top_crop_ratio)), :] = 0
    mask = fill_floor_mask_continuous(mask)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        bottom_band = int(h * 0.9)
        kept = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            touches_bottom = (y + ch) >= bottom_band
            centroid_y = y + ch / 2.0
            area = cv2.contourArea(contour)
            in_lower_scene = centroid_y >= h * 0.32
            if area > (h * w * 0.004) and (touches_bottom or in_lower_scene):
                kept.append(contour)
        if not kept:
            kept = [max(contours, key=cv2.contourArea)]
        cleaned = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cleaned, kept, -1, 255, thickness=-1)
        mask = fill_floor_mask_continuous(cleaned)

    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    return mask


def refine_floor_remove_foreground_objects(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Quita cojines/muebles por color; evita baldosas marrones (baja saturacion)."""
    h, w = mask.shape[:2]
    active = mask > 127
    if int(active.sum()) < 200:
        return mask

    y_sample = int(h * 0.68)
    sample_mask = active[y_sample:, :]
    pixels = img_bgr[y_sample:, :][sample_mask]
    if len(pixels) < 80:
        return mask

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    floor_lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    median = np.median(floor_lab, axis=0)
    diff = np.linalg.norm(lab_img - median, axis=2)

    y_idx = np.arange(h, dtype=np.float32)[:, None]
    peel_zone = y_idx < (h * 0.8)
    threshold = float(np.clip(np.percentile(diff[active], 60) + 14, 20, 38))
    peel = active & peel_zone & (diff > threshold) & (sat > 45)

    cleaned = mask.copy()
    cleaned[peel] = 0
    return fill_floor_mask_continuous(cleaned)


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
