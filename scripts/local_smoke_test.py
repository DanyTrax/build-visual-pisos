import base64
import io
import json
import sys
from pathlib import Path

import requests
from PIL import Image


def make_test_image_bytes() -> bytes:
    img = Image.new("RGB", (640, 420), color=(210, 210, 210))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def run(base_url: str = "http://localhost:8000") -> None:
    print("1) Health")
    health = requests.get(f"{base_url}/health", timeout=20)
    health.raise_for_status()
    print("   ", health.json())

    print("2) Login admin bootstrap")
    login = requests.post(
        f"{base_url}/api/admin/auth/login",
        json={"email": "admin@example.com", "password": "ChangeMe123!"},
        timeout=20,
    )
    login.raise_for_status()
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("   OK")

    print("3) Get textures")
    textures = requests.get(f"{base_url}/api/textures", timeout=20)
    textures.raise_for_status()
    textures_data = textures.json().get("textures", [])
    if not textures_data:
        raise RuntimeError("No hay texturas activas para la prueba")
    texture_id = textures_data[0]["id"]
    print(f"   Primera textura: {texture_id}")

    print("4) Analyze")
    test_img = make_test_image_bytes()
    analyze = requests.post(
        f"{base_url}/api/analyze",
        files={"image": ("test.jpg", test_img, "image/jpeg")},
        timeout=120,
    )
    analyze.raise_for_status()
    analyze_data = analyze.json()
    session_id = analyze_data["session_id"]
    print(f"   Session: {session_id}")

    print("5) Visualize")
    visualize = requests.post(
        f"{base_url}/api/visualize",
        json={"session_id": session_id, "texture_id": texture_id},
        timeout=120,
    )
    visualize.raise_for_status()
    result_b64 = visualize.json()["result_image_base64"]
    out_path = Path("local_test_result.jpg")
    out_path.write_bytes(base64.b64decode(result_b64))
    print(f"   Imagen generada: {out_path.resolve()}")

    print("6) List users (RBAC)")
    users = requests.get(f"{base_url}/api/admin/users", headers=headers, timeout=20)
    users.raise_for_status()
    print("   ", json.dumps(users.json(), indent=2))
    print("\nSmoke test completado.")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    run(url)
