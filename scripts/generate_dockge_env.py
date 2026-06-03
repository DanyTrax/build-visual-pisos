#!/usr/bin/env python3
"""
Generador interactivo de .env y comandos SSH para desplegar en Dockge.

Uso:
  python3 scripts/generate_dockge_env.py
  python3 scripts/generate_dockge_env.py --output .env --server-ip 192.168.1.50
"""

from __future__ import annotations

import argparse
import getpass
import secrets
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


def prompt_yes_no(text: str, default: bool = True) -> bool:
    default_label = "S/n" if default else "s/N"
    value = input(f"{text} ({default_label}): ").strip().lower()
    if not value:
        return default
    return value in {"s", "si", "y", "yes", "1", "true"}


def build_env_content(
    replicate_token: str,
    jwt_secret: str,
    admin_email: str,
    admin_password: str,
    api_port: str,
    web_port: str,
    server_ip: str,
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
JWT_SECRET={jwt_secret}
JWT_EXPIRES_MIN=720

ADMIN_BOOTSTRAP_EMAIL={admin_email}
ADMIN_BOOTSTRAP_PASSWORD={admin_password}

DATA_DIR=/app/data
API_PORT={api_port}
WEB_PORT={web_port}
"""


def build_ssh_script(stack_path: str) -> str:
    return f"""#!/bin/bash
# Ejecutar en el servidor Dockge (como root o con sudo)

set -e
STACK="{stack_path}"

mkdir -p "$STACK"
cd "$STACK"

if [ ! -d .git ]; then
  git init
  git remote add origin {REPO_URL} 2>/dev/null || git remote set-url origin {REPO_URL}
  git fetch origin
  git reset --hard origin/main
else
  git fetch origin
  git reset --hard origin/main
fi

if [ ! -f .env ]; then
  echo "ERROR: falta .env en $STACK"
  echo "Copia el .env generado desde tu PC:"
  echo "  scp .env root@TU_SERVIDOR:$STACK/.env"
  exit 1
fi

echo "Listo. Ahora en Dockge: Desplegar stack build-visual-pisos"
"""


def build_scp_command(local_env: Path, stack_path: str, server_user: str, server_host: str) -> str:
    return f'scp "{local_env.resolve()}" {server_user}@{server_host}:{stack_path}/.env'


def main() -> int:
    parser = argparse.ArgumentParser(description="Generador .env + guia Dockge para build-visual-pisos")
    parser.add_argument("--output", default=".env", help="Archivo .env local a generar (default: .env)")
    parser.add_argument("--stack-path", default=DEFAULT_STACK_PATH, help="Ruta del stack en el servidor")
    parser.add_argument("--server-ip", default="", help="IP o dominio del servidor")
    parser.add_argument("--server-user", default="root", help="Usuario SSH (default: root)")
    parser.add_argument("--api-port", default="8001", help="Puerto host API (default: 8001)")
    parser.add_argument("--web-port", default="8080", help="Puerto host web (default: 8080)")
    parser.add_argument("--non-interactive", action="store_true", help="Usar defaults sin preguntas")
    args = parser.parse_args()

    print("=== Generador Dockge: build-visual-pisos ===\n")

    if args.non_interactive:
        replicate_token = ""
        admin_email = "admin@example.com"
        admin_password = "ChangeMe123!"
        jwt_secret = secrets.token_urlsafe(48)
        server_ip = args.server_ip or "TU_IP"
    else:
        replicate_token = prompt("Token Replicate (r8_...)", "")
        if not replicate_token:
            print("AVISO: sin token, la IA usara fallback heuristico en pruebas.")

        admin_email = prompt("Email admin bootstrap", "admin@example.com")
        admin_password = getpass.getpass("Password admin bootstrap: ")
        if len(admin_password) < 8:
            print("AVISO: password corto; usa minimo 8 caracteres en produccion.")

        auto_jwt = prompt_yes_no("Generar JWT_SECRET automaticamente?", True)
        jwt_secret = secrets.token_urlsafe(48) if auto_jwt else prompt("JWT_SECRET", secrets.token_urlsafe(48))

        server_ip = args.server_ip or prompt("IP o dominio del servidor", "TU_IP")

    api_port = args.api_port
    web_port = args.web_port
    if not args.non_interactive:
        api_port = prompt("Puerto API en host", args.api_port)
        web_port = prompt("Puerto WEB en host", args.web_port)

    env_content = build_env_content(
        replicate_token=replicate_token,
        jwt_secret=jwt_secret,
        admin_email=admin_email.lower(),
        admin_password=admin_password,
        api_port=api_port,
        web_port=web_port,
        server_ip=server_ip,
    )

    output_path = Path(args.output)
    output_path.write_text(env_content, encoding="utf-8")
    print(f"\nOK: archivo generado -> {output_path.resolve()}")

    ssh_script_path = output_path.parent / "server_setup.sh"
    ssh_script_path.write_text(build_ssh_script(args.stack_path), encoding="utf-8")
    ssh_script_path.chmod(0o755)
    print(f"OK: script servidor -> {ssh_script_path.resolve()}")

    print("\n--- Proximos pasos ---")
    print("1) En tu PC (ya tienes .env generado).")
    if server_ip and server_ip != "TU_IP":
        print(f"2) Subir .env al servidor:")
        print(f"   {build_scp_command(output_path, args.stack_path, args.server_user, server_ip)}")
        print("3) En SSH del servidor:")
        print(f"   bash {args.stack_path}/server_setup.sh")
        print("   (o copia y ejecuta el contenido de server_setup.sh)")
    else:
        print("2) Sube .env al servidor con scp (reemplaza TU_IP).")
        print("3) Ejecuta server_setup.sh en el servidor.")

    print("4) En Dockge: stack build-visual-pisos -> Desplegar")

    print("\n--- URLs ---")
    host = server_ip if server_ip != "TU_IP" else "TU_IP"
    print(f"Visualizer: http://{host}:{web_port}/")
    print(f"Admin:      http://{host}:{web_port}/admin/")
    print(f"Health API: http://{host}:{api_port}/health")
    print(f"Login admin: {admin_email}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
