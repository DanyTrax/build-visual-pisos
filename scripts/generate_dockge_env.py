#!/usr/bin/env python3
"""
Generador de .env para Dockge.
Las llaves JWT y password admin se generan automaticamente dentro de Docker
en el primer arranque (data/secrets.json).

Uso:
  python3 scripts/generate_dockge_env.py
  python3 scripts/generate_dockge_env.py --server-ip 192.168.1.50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_URL = "https://github.com/DanyTrax/build-visual-pisos.git"
DEFAULT_STACK_PATH = "/opt/stacks/build-visual-pisos"
DEFAULT_MODEL = (
    "schananas/grounded_sam:"
    "ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c"
)


def prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or default


def build_env_content(
    replicate_token: str,
    admin_email: str,
    api_port: str,
    web_port: str,
) -> str:
    return f"""REPLICATE_API_TOKEN={replicate_token}

REPLICATE_MODEL={DEFAULT_MODEL}
FLOOR_TEXT_PROMPT=floor, ground, flooring
NEGATIVE_MASK_PROMPT=furniture, wall, person, rug
DETECTION_THRESHOLD=0.30
BOX_THRESHOLD=0.25
MAX_IMAGE_WIDTH=1280
ENABLE_FALLBACK_HEURISTIC=true
MASK_FEATHER_PX=11
BLEND_STRENGTH=0.75

ALLOWED_ORIGINS=*
JWT_EXPIRES_MIN=720

# Llaves random: las genera Docker al primer arranque en data/secrets.json
AUTO_GENERATE_SECRETS=true
ADMIN_BOOTSTRAP_EMAIL={admin_email}

DATA_DIR=/app/data
API_PORT={api_port}
WEB_PORT={web_port}
"""


def build_ssh_script(stack_path: str) -> str:
    return f"""#!/bin/bash
set -e
STACK="{stack_path}"
mkdir -p "$STACK"
cd "$STACK"

if [ ! -d .git ]; then
  git init
  git remote add origin {REPO_URL} 2>/dev/null || git remote set-url origin {REPO_URL}
fi
git fetch origin
git reset --hard origin/main

if [ ! -f .env ]; then
  echo "Falta .env. Sube el generado desde tu PC con scp."
  exit 1
fi

echo "Repo listo. En Dockge pulsa Desplegar."
echo "Tras el primer deploy, lee la clave admin con:"
echo "  cat {stack_path}/data/secrets.json"
"""


def build_scp_command(local_env: Path, stack_path: str, server_user: str, server_host: str) -> str:
    return f'scp "{local_env.resolve()}" {server_user}@{server_host}:{stack_path}/.env'


def main() -> int:
    parser = argparse.ArgumentParser(description="Generador .env Dockge (llaves random en Docker)")
    parser.add_argument("--output", default=".env")
    parser.add_argument("--stack-path", default=DEFAULT_STACK_PATH)
    parser.add_argument("--server-ip", default="")
    parser.add_argument("--server-user", default="root")
    parser.add_argument("--api-port", default="8001")
    parser.add_argument("--web-port", default="8080")
    parser.add_argument("--non-interactive", action="store_true")
    args = parser.parse_args()

    print("=== Generador Dockge (llaves random en Docker) ===\n")

    if args.non_interactive:
        replicate_token = ""
        admin_email = "admin@example.com"
        server_ip = args.server_ip or "TU_IP"
        api_port = args.api_port
        web_port = args.web_port
    else:
        replicate_token = prompt("Token Replicate (r8_...)", "")
        admin_email = prompt("Email admin bootstrap", "admin@example.com")
        server_ip = args.server_ip or prompt("IP o dominio del servidor", "TU_IP")
        api_port = prompt("Puerto API host", args.api_port)
        web_port = prompt("Puerto WEB host", args.web_port)

    output_path = Path(args.output)
    output_path.write_text(
        build_env_content(replicate_token, admin_email.lower(), api_port, web_port),
        encoding="utf-8",
    )
    print(f"OK: {output_path.resolve()}")

    ssh_script_path = output_path.parent / "server_setup.sh"
    ssh_script_path.write_text(build_ssh_script(args.stack_path), encoding="utf-8")
    ssh_script_path.chmod(0o755)
    print(f"OK: {ssh_script_path.resolve()}")

    print("\n--- Pasos ---")
    print("1) Sube .env al servidor (scp).")
    if server_ip != "TU_IP":
        print(f"   {build_scp_command(output_path, args.stack_path, args.server_user, server_ip)}")
    print("2) SSH: bash server_setup.sh")
    print("3) Dockge -> Desplegar")
    print("4) Obtener password admin generado:")
    print(f"   cat {args.stack_path}/data/secrets.json")

    host = server_ip if server_ip != "TU_IP" else "TU_IP"
    print("\n--- URLs ---")
    print(f"Visualizer: http://{host}:{web_port}/")
    print(f"Admin:      http://{host}:{web_port}/admin/")
    print(f"Health:     http://{host}:{api_port}/health")
    print(f"Login email: {admin_email}")
    print("Login pass:  (ver data/secrets.json en el servidor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
