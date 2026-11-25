import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
import pystray
import uvicorn

# --- Rutas portables basadas en la ubicación del script o ejecutable ---
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

APP_DIR = BASE_DIR / "app"
STATIC_DIR = APP_DIR / "static"
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "datos"
LOGS_DIR = BASE_DIR / "logs"

for folder in (STATIC_DIR, ASSETS_DIR, DATA_DIR, LOGS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# --- Logging en UTF-8 (esencial sin consola visible) ---
log_file = LOGS_DIR / "myteacher.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("myTeacher")


def find_free_port(host: str = "127.0.0.1", preferred_port: int = 8765) -> int:
    """Devuelve preferred_port si está libre o un puerto efímero disponible."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, preferred_port))
            return preferred_port
        except OSError:
            s.bind((host, 0))
            return s.getsockname()[1]


def build_tray_image() -> Image.Image:
    """Carga assets/icon.ico si existe; si no, genera un icono simple en memoria."""
    ico_path = ASSETS_DIR / "icon.ico"
    if ico_path.exists():
        try:
            return Image.open(ico_path).copy()
        except Exception as exc:
            logger.warning("No se pudo cargar icon.ico, usando icono por defecto: %s", exc)

    img = Image.new("RGBA", (64, 64), (24, 28, 36, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((8, 8, 56, 56), radius=12, fill=(37, 99, 235, 255))
    draw.text((21, 20), "mT", fill=(255, 255, 255, 255))
    return img


def create_app() -> FastAPI:
    """Crea la aplicación FastAPI de myTeacher."""
    web_app = FastAPI(title="myTeacher", version="0.1.0")

    if STATIC_DIR.exists():
        web_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @web_app.get("/", response_class=FileResponse)
    def root():
        index_path = STATIC_DIR / "index.html"
        return FileResponse(index_path, media_type="text/html; charset=utf-8")

    @web_app.get("/health")
    def health():
        return JSONResponse({"status": "ok", "app": "myTeacher"})

    return web_app


def wait_until_ready(url: str, timeout: float = 10.0) -> bool:
    """Espera hasta que el servidor local responda en /health."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.15)
    return False


class MyTeacherController:
    """Coordina el servidor Uvicorn en segundo plano y el icono de System Tray."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.fastapi_app = create_app()
        self.config = uvicorn.Config(
            app=self.fastapi_app,
            host=self.host,
            port=self.port,
            log_config=None,
            access_log=False,
        )
        self.server = uvicorn.Server(self.config)
        self.server_thread: threading.Thread | None = None
        self.icon: pystray.Icon | None = None

    def _run_server(self) -> None:
        logger.info("Iniciando servidor myTeacher en %s", self.base_url)
        try:
            self.server.run()
        except Exception:
            logger.exception("Error fatal en el servidor Uvicorn")

    def open_browser(self, icon=None, item=None) -> None:
        logger.info("Abriendo navegador en %s", self.base_url)
        webbrowser.open(self.base_url, new=2)

    def quit_app(self, icon: pystray.Icon, item=None) -> None:
        logger.info("Solicitado cierre de myTeacher desde System Tray.")
        self.server.should_exit = True
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5.0)
        icon.stop()
        logger.info("myTeacher cerrado limpiamente.")

    def run(self) -> None:
        self.server_thread = threading.Thread(target=self._run_server, name="UvicornThread", daemon=False)
        self.server_thread.start()

        if wait_until_ready(f"{self.base_url}/health"):
            self.open_browser()
        else:
            logger.warning("El servidor tardó más de lo previsto en responder; intentando abrir navegador.")
            self.open_browser()

        menu = pystray.Menu(
            pystray.MenuItem("Abrir myTeacher", self.open_browser, default=True),
            pystray.MenuItem("Salir", self.quit_app),
        )
        self.icon = pystray.Icon("myTeacher", build_tray_image(), "myTeacher", menu)
        self.icon.run()


def main() -> None:
    port = find_free_port("127.0.0.1", 8765)
    controller = MyTeacherController(host="127.0.0.1", port=port)
    controller.run()


if __name__ == "__main__":
    main()