import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .auth import create_access_token, decode_access_token, hash_password, verify_password
from .config import get_settings
from .image_ops import (
    blend_floor,
    create_overlay,
    encode_image_base64,
    load_image_from_bytes,
    make_tiled_texture,
    perspective_texture,
)
from .segmentation import segment_floor_with_replicate
from .storage import (
    cleanup_sessions,
    create_session_folder,
    ensure_data_layout,
    guess_extension,
    new_session_id,
    read_ai_config,
    read_catalog,
    read_users,
    session_dir,
    utc_now,
    write_ai_config,
    write_catalog,
    write_users,
)

settings = get_settings()
app = FastAPI(title="Floor Visualizer API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeResponse(BaseModel):
    session_id: str
    floor_detected: bool
    message: str
    mask_preview_base64: str


class VisualizeRequest(BaseModel):
    session_id: str
    texture_id: str


class LoginRequest(BaseModel):
    email: str
    password: str


class UpdateAiConfigRequest(BaseModel):
    replicate_model: str
    floor_text_prompt: str
    negative_mask_prompt: str = ""
    objects_subtraction_prompt: str = ""
    enable_object_subtraction: bool = True
    detection_threshold: float = Field(ge=0, le=1)
    box_threshold: float = Field(ge=0, le=1)
    max_image_width: int = Field(ge=512, le=4096)
    enable_fallback_heuristic: bool
    mask_feather_px: int = Field(ge=3, le=99)
    mask_adjustment_factor: int = Field(ge=0, le=50)
    blend_strength: float = Field(ge=0, le=1)


class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: str
    active: bool = True


class UpdateUserRequest(BaseModel):
    role: str | None = None
    active: bool | None = None


class ResetPasswordRequest(BaseModel):
    password: str


def _effective_ai_config() -> dict:
    config = read_ai_config(settings.data_dir)
    defaults = {
        "replicate_model": settings.replicate_model,
        "floor_text_prompt": settings.floor_text_prompt,
        "negative_mask_prompt": settings.negative_mask_prompt,
        "detection_threshold": settings.detection_threshold,
        "box_threshold": settings.box_threshold,
        "max_image_width": settings.max_image_width,
        "enable_fallback_heuristic": settings.enable_fallback_heuristic,
        "mask_feather_px": settings.mask_feather_px,
        "mask_adjustment_factor": settings.mask_adjustment_factor,
        "blend_strength": settings.blend_strength,
    }
    defaults.update(config)
    return defaults


def _public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "email": user["email"],
        "role": user["role"],
        "active": user.get("active", True),
        "created_at": user.get("created_at"),
    }


def _require_auth(authorization: str = Header(default="")) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token no enviado")
    token = authorization.replace("Bearer ", "", 1)
    try:
        payload = decode_access_token(token, settings.jwt_secret)
        return {"email": payload["sub"], "role": payload["role"]}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail=f"Token invalido: {exc}") from exc


def _require_role(*allowed_roles: str):
    def checker(current=Depends(_require_auth)):
        if current["role"] not in allowed_roles:
            raise HTTPException(status_code=403, detail="Permisos insuficientes")
        return current

    return checker


def _bootstrap_admin_if_needed() -> None:
    users_data = read_users(settings.data_dir)
    if users_data.get("users"):
        return
    first = {
        "id": "u-admin-1",
        "email": settings.admin_bootstrap_email.lower(),
        "password_hash": hash_password(settings.admin_bootstrap_password),
        "role": "admin",
        "active": True,
        "created_at": utc_now().isoformat(),
    }
    users_data["users"] = [first]
    write_users(settings.data_dir, users_data)


@app.on_event("startup")
async def startup() -> None:
    try:
        ensure_data_layout(settings.data_dir)
        _bootstrap_admin_if_needed()
        print("[startup] API lista")
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] ERROR: {exc}")
        raise


@app.get("/health")
async def health() -> dict:
    data_writable = os.access(settings.data_dir, os.W_OK)
    secrets_file = Path(settings.data_dir) / "secrets.json"
    return {
        "status": "ok",
        "service": "floor-visualizer-api",
        "replicate_configured": bool(settings.replicate_api_token),
        "data_writable": data_writable,
        "secrets_file_exists": secrets_file.exists(),
        "auto_generate_secrets": True,
    }


@app.get("/api/textures")
async def list_public_textures() -> dict:
    catalog = read_catalog(settings.data_dir)
    textures = [t for t in catalog.get("textures", []) if t.get("active", True)]
    textures.sort(key=lambda x: x.get("sort_order", 9999))
    for texture in textures:
        texture["image_url"] = f"/static/textures/{texture['filename']}"
    return {"textures": textures}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_room(image: UploadFile = File(...)) -> AnalyzeResponse:
    cleanup_sessions(settings.data_dir, ttl_minutes=30)
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Imagen vacia")
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Imagen supera 10MB")

    cfg = _effective_ai_config()
    img_bgr = load_image_from_bytes(raw, max_width=int(cfg["max_image_width"]))
    mask, message = segment_floor_with_replicate(raw, img_bgr, cfg, settings.replicate_api_token)

    session_id = new_session_id()
    folder = create_session_folder(settings.data_dir, session_id)
    cv2.imwrite(str(folder / "original.jpg"), img_bgr)
    cv2.imwrite(str(folder / "mask.png"), mask)
    (folder / "meta.json").write_text(
        f'{{"created_at":"{datetime.now(timezone.utc).isoformat()}","message":"{message}"}}', encoding="utf-8"
    )

    overlay = create_overlay(img_bgr, mask)
    return AnalyzeResponse(
        session_id=session_id,
        floor_detected=bool((mask > 20).sum() > 100),
        message=message,
        mask_preview_base64=encode_image_base64(overlay, quality=88),
    )


@app.post("/api/visualize")
async def visualize_floor(payload: VisualizeRequest) -> dict:
    folder = session_dir(settings.data_dir, payload.session_id)
    if not folder.exists():
        raise HTTPException(status_code=404, detail="Sesion no encontrada o expirada")

    catalog = read_catalog(settings.data_dir)
    texture = next((t for t in catalog.get("textures", []) if t.get("id") == payload.texture_id and t.get("active", True)), None)
    if not texture:
        raise HTTPException(status_code=404, detail="Textura no encontrada")

    original = cv2.imread(str(folder / "original.jpg"))
    mask = cv2.imread(str(folder / "mask.png"), cv2.IMREAD_GRAYSCALE)
    if original is None or mask is None:
        raise HTTPException(status_code=500, detail="Sesion invalida")

    texture_path = Path(settings.data_dir, "textures", texture["filename"])
    texture_img = cv2.imread(str(texture_path))
    if texture_img is None:
        raise HTTPException(status_code=500, detail="No se pudo leer la textura")

    cfg = _effective_ai_config()
    tiled = make_tiled_texture(texture_img, original.shape[:2])
    warped = perspective_texture(tiled, mask)
    result = blend_floor(
        original_bgr=original,
        floor_texture_bgr=warped,
        mask=mask,
        feather_px=int(cfg.get("mask_feather_px", 11)),
        strength=float(cfg.get("blend_strength", 0.75)),
    )
    return {"result_image_base64": encode_image_base64(result, quality=90)}


@app.post("/api/admin/auth/login")
async def admin_login(payload: LoginRequest) -> dict:
    users_data = read_users(settings.data_dir)
    target = next((u for u in users_data.get("users", []) if u["email"] == payload.email.lower()), None)
    if not target or not target.get("active", True):
        raise HTTPException(status_code=401, detail="Credenciales invalidas")
    if not verify_password(payload.password, target["password_hash"]):
        raise HTTPException(status_code=401, detail="Credenciales invalidas")
    token = create_access_token(target["email"], target["role"], settings.jwt_secret, settings.jwt_expires_min)
    return {"access_token": token, "token_type": "bearer", "user": _public_user(target)}


@app.get("/api/admin/me")
async def admin_me(current=Depends(_require_role("admin", "editor", "viewer"))) -> dict:
    users_data = read_users(settings.data_dir)
    target = next((u for u in users_data.get("users", []) if u["email"] == current["email"]), None)
    if not target:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return {"user": _public_user(target)}


@app.get("/api/admin/textures")
async def admin_textures_list(current=Depends(_require_role("admin", "editor", "viewer"))) -> dict:
    return read_catalog(settings.data_dir)


@app.post("/api/admin/textures")
async def admin_texture_create(
    name: str = Form(...),
    category: str = Form("general"),
    sort_order: int = Form(999),
    active: bool = Form(True),
    image: UploadFile = File(...),
    current=Depends(_require_role("admin", "editor")),
) -> dict:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Imagen vacia")
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Imagen supera 5MB")
    ext = guess_extension(image.filename or "texture.jpg")
    slug = f"{name.strip().lower().replace(' ', '-')}-{new_session_id()[:8]}"
    filename = f"{slug}{ext}"
    target = Path(settings.data_dir, "textures", filename)
    target.write_bytes(raw)

    catalog = read_catalog(settings.data_dir)
    item = {
        "id": slug,
        "name": name,
        "category": category,
        "filename": filename,
        "active": active,
        "sort_order": sort_order,
        "created_at": utc_now().isoformat(),
        "created_by": current["email"],
    }
    catalog.setdefault("textures", []).append(item)
    write_catalog(settings.data_dir, catalog)
    return {"texture": item}


@app.put("/api/admin/textures/{texture_id}")
async def admin_texture_update(texture_id: str, payload: dict, current=Depends(_require_role("admin", "editor"))) -> dict:
    catalog = read_catalog(settings.data_dir)
    target = next((t for t in catalog.get("textures", []) if t["id"] == texture_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Textura no encontrada")
    for field in ("name", "category", "sort_order", "active"):
        if field in payload:
            target[field] = payload[field]
    target["updated_at"] = utc_now().isoformat()
    target["updated_by"] = current["email"]
    write_catalog(settings.data_dir, catalog)
    return {"texture": target}


@app.delete("/api/admin/textures/{texture_id}")
async def admin_texture_delete(texture_id: str, hard: bool = False, current=Depends(_require_role("admin", "editor"))) -> dict:
    catalog = read_catalog(settings.data_dir)
    target = next((t for t in catalog.get("textures", []) if t["id"] == texture_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Textura no encontrada")
    if hard:
        try:
            Path(settings.data_dir, "textures", target["filename"]).unlink(missing_ok=True)
        except OSError:
            pass
        catalog["textures"] = [t for t in catalog.get("textures", []) if t["id"] != texture_id]
    else:
        target["active"] = False
    write_catalog(settings.data_dir, catalog)
    return {"ok": True}


@app.get("/api/admin/ai-config")
async def admin_get_ai_config(current=Depends(_require_role("admin"))) -> dict:
    cfg = _effective_ai_config()
    cfg["replicate_token_configured"] = bool(settings.replicate_api_token)
    return cfg


@app.put("/api/admin/ai-config")
async def admin_put_ai_config(payload: UpdateAiConfigRequest, current=Depends(_require_role("admin"))) -> dict:
    write_ai_config(settings.data_dir, payload.model_dump())
    return {"ok": True, "config": payload.model_dump()}


@app.post("/api/admin/ai-config/test")
async def admin_ai_config_test(
    image: UploadFile = File(...),
    current=Depends(_require_role("admin")),
) -> dict:
    raw = await image.read()
    cfg = _effective_ai_config()
    img = load_image_from_bytes(raw, max_width=int(cfg["max_image_width"]))
    mask, message = segment_floor_with_replicate(raw, img, cfg, settings.replicate_api_token)
    overlay = create_overlay(img, mask)
    return {"message": message, "preview_base64": encode_image_base64(overlay, quality=88)}


@app.get("/api/admin/users")
async def admin_users_list(current=Depends(_require_role("admin"))) -> dict:
    users_data = read_users(settings.data_dir)
    return {"users": [_public_user(u) for u in users_data.get("users", [])]}


@app.post("/api/admin/users")
async def admin_users_create(payload: CreateUserRequest, current=Depends(_require_role("admin"))) -> dict:
    role = payload.role.lower()
    if role not in {"admin", "editor", "viewer"}:
        raise HTTPException(status_code=400, detail="Rol invalido")
    users_data = read_users(settings.data_dir)
    if any(u["email"] == payload.email.lower() for u in users_data.get("users", [])):
        raise HTTPException(status_code=409, detail="Email ya existe")
    user = {
        "id": f"u-{new_session_id()[:10]}",
        "email": payload.email.lower(),
        "password_hash": hash_password(payload.password),
        "role": role,
        "active": payload.active,
        "created_at": utc_now().isoformat(),
    }
    users_data.setdefault("users", []).append(user)
    write_users(settings.data_dir, users_data)
    return {"user": _public_user(user)}


@app.put("/api/admin/users/{user_id}")
async def admin_users_update(user_id: str, payload: UpdateUserRequest, current=Depends(_require_role("admin"))) -> dict:
    users_data = read_users(settings.data_dir)
    target = next((u for u in users_data.get("users", []) if u["id"] == user_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    if payload.role:
        if payload.role.lower() not in {"admin", "editor", "viewer"}:
            raise HTTPException(status_code=400, detail="Rol invalido")
        target["role"] = payload.role.lower()
    if payload.active is not None:
        target["active"] = payload.active
    target["updated_at"] = utc_now().isoformat()
    write_users(settings.data_dir, users_data)
    return {"user": _public_user(target)}


@app.post("/api/admin/users/{user_id}/reset-password")
async def admin_users_reset_password(user_id: str, payload: ResetPasswordRequest, current=Depends(_require_role("admin"))) -> dict:
    users_data = read_users(settings.data_dir)
    target = next((u for u in users_data.get("users", []) if u["id"] == user_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    target["password_hash"] = hash_password(payload.password)
    target["updated_at"] = utc_now().isoformat()
    write_users(settings.data_dir, users_data)
    return {"ok": True}
