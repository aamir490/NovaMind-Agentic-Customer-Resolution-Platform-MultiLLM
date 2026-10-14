from pathlib import Path
import os

auth_content = os.environ.get("NOVAMIND_AUTH_CONTENT")
if not auth_content:
    raise RuntimeError("NOVAMIND_AUTH_CONTENT is required")

auth_dir = Path("/dev/shm/novamind")
auth_dir.mkdir(parents=True, exist_ok=True)

auth_file = auth_dir / "auth.json"
auth_file.write_text(auth_content, encoding="utf-8")

os.environ["NOVAMIND_AUTH_FILE"] = str(auth_file)

os.execvp(
    "python",
    [
        "python",
        "-m",
        "uvicorn",
        "backend.container.bootstrap:create_container_app",
        "--factory",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--workers",
        "1",
        "--no-proxy-headers",
        "--no-access-log",
    ],
)
