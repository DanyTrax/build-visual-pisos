import json
import secrets
from datetime import datetime, timezone
from pathlib import Path


PLACEHOLDER_VALUES = {
    "",
    "change-this-secret",
    "auto",
    "generate",
    "changeme123!",
    "changeme",
}


def _secrets_path(data_dir: str) -> Path:
    return Path(data_dir) / "secrets.json"


def ensure_runtime_secrets(data_dir: str, force_new: bool = False) -> dict:
    """
    Genera y persiste llaves aleatorias en data/secrets.json (volumen Docker).
    Si el archivo ya existe, reutiliza las mismas llaves entre reinicios.
    """
    path = _secrets_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force_new:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    payload = {
        "jwt_secret": secrets.token_urlsafe(48),
        "admin_bootstrap_password": secrets.token_urlsafe(16),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass

    print("[secrets] Llaves generadas automaticamente en data/secrets.json")
    print(f"[secrets] Admin bootstrap password: {payload['admin_bootstrap_password']}")
    print("[secrets] Guarda esa clave; no se vuelve a mostrar en logs posteriores.")
    return payload


def is_placeholder(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in PLACEHOLDER_VALUES
