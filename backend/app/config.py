import os
from dataclasses import dataclass

from .secrets_manager import ensure_runtime_secrets, is_placeholder


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    replicate_api_token: str
    replicate_model: str
    floor_text_prompt: str
    negative_mask_prompt: str
    detection_threshold: float
    box_threshold: float
    max_image_width: int
    enable_fallback_heuristic: bool
    mask_feather_px: int
    mask_adjustment_factor: int
    blend_strength: float
    allowed_origins: list[str]
    jwt_secret: str
    jwt_expires_min: int
    admin_bootstrap_email: str
    admin_bootstrap_password: str
    data_dir: str


def get_settings() -> Settings:
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
    origins = ["*"] if allowed_origins == "*" else [o.strip() for o in allowed_origins.split(",") if o.strip()]

    data_dir = os.getenv("DATA_DIR", "/app/data")
    auto_generate = _to_bool(os.getenv("AUTO_GENERATE_SECRETS", "true"), True)

    jwt_secret = os.getenv("JWT_SECRET", "")
    admin_password = os.getenv("ADMIN_BOOTSTRAP_PASSWORD", "")

    if auto_generate or is_placeholder(jwt_secret) or is_placeholder(admin_password):
        runtime = ensure_runtime_secrets(data_dir)
        if auto_generate or is_placeholder(jwt_secret):
            jwt_secret = runtime["jwt_secret"]
        if auto_generate or is_placeholder(admin_password):
            admin_password = runtime["admin_bootstrap_password"]

    return Settings(
        replicate_api_token=os.getenv("REPLICATE_API_TOKEN", ""),
        replicate_model=os.getenv(
            "REPLICATE_MODEL",
            "schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c",
        ),
        floor_text_prompt=os.getenv("FLOOR_TEXT_PROMPT", "floor, ground, flooring"),
        negative_mask_prompt=os.getenv("NEGATIVE_MASK_PROMPT", "furniture, wall, person, rug"),
        detection_threshold=float(os.getenv("DETECTION_THRESHOLD", "0.30")),
        box_threshold=float(os.getenv("BOX_THRESHOLD", "0.25")),
        max_image_width=int(os.getenv("MAX_IMAGE_WIDTH", "1280")),
        enable_fallback_heuristic=_to_bool(os.getenv("ENABLE_FALLBACK_HEURISTIC", "true"), True),
        mask_feather_px=int(os.getenv("MASK_FEATHER_PX", "11")),
        mask_adjustment_factor=int(os.getenv("MASK_ADJUSTMENT_FACTOR", "12")),
        blend_strength=float(os.getenv("BLEND_STRENGTH", "0.75")),
        allowed_origins=origins,
        jwt_secret=jwt_secret,
        jwt_expires_min=int(os.getenv("JWT_EXPIRES_MIN", "720")),
        admin_bootstrap_email=os.getenv("ADMIN_BOOTSTRAP_EMAIL", "admin@example.com"),
        admin_bootstrap_password=admin_password,
        data_dir=data_dir,
    )
