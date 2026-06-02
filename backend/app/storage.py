import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock


_catalog_lock = Lock()
_ai_lock = Lock()
_users_lock = Lock()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_data_layout(data_dir: str) -> None:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(data_dir, "textures").mkdir(parents=True, exist_ok=True)
    Path(data_dir, "sessions").mkdir(parents=True, exist_ok=True)
    for filename, default in (
        ("catalog.json", {"textures": []}),
        ("ai_config.json", {}),
        ("users.json", {"users": []}),
    ):
        target = Path(data_dir, filename)
        if not target.exists():
            target.write_text(json.dumps(default, indent=2), encoding="utf-8")


def _read_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_catalog(data_dir: str) -> dict:
    return _read_json(Path(data_dir, "catalog.json"), {"textures": []})


def write_catalog(data_dir: str, payload: dict) -> None:
    with _catalog_lock:
        Path(data_dir, "catalog.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_ai_config(data_dir: str) -> dict:
    return _read_json(Path(data_dir, "ai_config.json"), {})


def write_ai_config(data_dir: str, payload: dict) -> None:
    with _ai_lock:
        Path(data_dir, "ai_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_users(data_dir: str) -> dict:
    return _read_json(Path(data_dir, "users.json"), {"users": []})


def write_users(data_dir: str, payload: dict) -> None:
    with _users_lock:
        Path(data_dir, "users.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def new_session_id() -> str:
    return uuid.uuid4().hex


def session_dir(data_dir: str, session_id: str) -> Path:
    return Path(data_dir, "sessions", session_id)


def create_session_folder(data_dir: str, session_id: str) -> Path:
    target = session_dir(data_dir, session_id)
    target.mkdir(parents=True, exist_ok=True)
    return target


def cleanup_sessions(data_dir: str, ttl_minutes: int = 30) -> None:
    sessions_path = Path(data_dir, "sessions")
    if not sessions_path.exists():
        return
    threshold = utc_now() - timedelta(minutes=ttl_minutes)
    for item in sessions_path.iterdir():
        if not item.is_dir():
            continue
        modified = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc)
        if modified < threshold:
            for child in item.iterdir():
                if child.is_file():
                    child.unlink(missing_ok=True)
            item.rmdir()


def guess_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".svg"}:
        return ext
    return ".jpg"
