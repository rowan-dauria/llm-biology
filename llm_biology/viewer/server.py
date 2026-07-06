"""Lightweight local server for Neuronpedia-compatible attribution graphs."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import copy
import gzip
import http.server
import json
import logging
import mimetypes
import re
import socketserver
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from llm_biology.attribution.circuit_graph_export import write_graph_metadata

logger = logging.getLogger(__name__)
logger.propagate = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "data" / "ui_graphs"
DEFAULT_STATIC_DIR = Path(__file__).parent / "static"
GZIP_MIN_BYTES = 1 << 20


class BiologyApp:
    """Application state for the viewer: where graphs, features, and static/frontend assets live."""

    def __init__(
        self,
        *,
        graph_file_dir: Path | str = DEFAULT_GRAPH_DIR,
        frontend_dir: Path | str | None = None,
        static_dir: Path | str = DEFAULT_STATIC_DIR,
    ) -> None:
        self.graph_file_dir = Path(graph_file_dir).resolve()
        self.frontend_dir = Path(frontend_dir).resolve() if frontend_dir else resolve_frontend_dir()
        self.static_dir = Path(static_dir).resolve()
        self.graph_file_dir.mkdir(parents=True, exist_ok=True)

    def shutdown(self) -> None:
        """No-op hook called by :meth:`Server.stop`; present for subclasses that hold resources."""
        return

    def upload_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate, normalize, and save an uploaded graph JSON; return its slug and URL."""
        raw_graph = payload.get("graph")
        if not isinstance(raw_graph, dict):
            raise RequestError(400, "graph must be an object")

        slug = upload_slug(
            raw_graph,
            slug_override=optional_string(payload, "slug"),
            filename=optional_string(payload, "filename"),
        )
        graph, metadata = normalize_uploaded_graph(raw_graph, slug)
        graph_path = self.graph_file_dir / f"{slug}.json"
        write_json(graph_path, graph)
        write_graph_metadata(self.graph_file_dir, metadata)

        return {
            "slug": slug,
            "graph_url": f"/graph_data/{slug}.json",
            "metadata": metadata,
        }


class RequestError(Exception):
    """An HTTP error to return to the client, carrying its intended status code."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class ReusableThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """A ``TCPServer`` that handles each request on its own thread and allows quick address reuse."""

    allow_reuse_address = True
    daemon_threads = True


class BiologyRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Routes GET/HEAD/POST requests to graph, feature, and frontend-asset serving."""

    server: BiologyTCPServer

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        server = args[2]
        super().__init__(*args, directory=str(server.app.static_dir), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Route the base handler's access log line through the module logger instead of stderr."""
        logger.info(
            "%s - - [%s] %s", self.address_string(), self.log_date_time_string(), format % args
        )

    def do_HEAD(self) -> None:
        self._dispatch(send_body=False)

    def do_GET(self) -> None:
        self._dispatch(send_body=True)

    def do_POST(self) -> None:
        self._dispatch(send_body=True)

    def _dispatch(self, *, send_body: bool) -> None:
        """Route the request through :meth:`_handle`, converting exceptions into JSON error responses."""
        try:
            self._handle(send_body=send_body)
        except RequestError as exc:
            self._send_json({"error": exc.message}, status=exc.status, send_body=send_body)
        except Exception as exc:
            logger.exception("Error handling %s %s: %s", self.command, self.path, exc)
            self._send_json({"error": "internal server error"}, status=500, send_body=send_body)

    def _handle(self, *, send_body: bool) -> None:
        """Dispatch one request by method and path prefix; raises :class:`RequestError` on 404/etc."""
        parsed = urlparse(self.path)
        path = parsed.path

        if self.command == "POST" and path == "/api/upload_graph":
            self._send_json(self.server.app.upload_graph(self._read_json_body()), status=201)
            return
        if self.command == "POST" and path.startswith("/save_graph/"):
            self._handle_save_graph(path, send_body=send_body)
            return
        if self.command in {"GET", "HEAD"} and path.startswith(("/data/", "/graph_data/")):
            rel = (
                path.removeprefix("/data/")
                if path.startswith("/data/")
                else path.removeprefix("/graph_data/")
            )
            if rel == "graph-metadata.json":
                self._serve_metadata(send_body=send_body)
            else:
                self._serve_file(
                    self.server.app.graph_file_dir,
                    rel,
                    send_body=send_body,
                    no_store=not rel.startswith("features/"),
                )
            return
        if self.command in {"GET", "HEAD"} and path.startswith("/features/"):
            self._serve_file(
                self.server.app.graph_file_dir / "features",
                path.removeprefix("/features/"),
                send_body=send_body,
            )
            return
        if self.command in {"GET", "HEAD"} and path.startswith("/ct/"):
            self._serve_file(
                self.server.app.frontend_dir,
                path.removeprefix("/ct/"),
                send_body=send_body,
            )
            return
        if self.command in {"GET", "HEAD"} and path in {"/", "/index.html"}:
            self._serve_file(
                self.server.app.static_dir, "index.html", send_body=send_body, no_store=True
            )
            return
        if self.command in {"GET", "HEAD"}:
            rel = path.lstrip("/")
            if rel in {"biology-server.js", "biology-server.css"}:
                self._serve_file(
                    self.server.app.static_dir, rel, send_body=send_body, no_store=True
                )
                return
        raise RequestError(404, "not found")

    def _read_json_body(self) -> dict[str, Any]:
        """Read and parse the request body as a JSON object; ``{}`` if empty, else raises on invalid JSON."""
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RequestError(400, "request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise RequestError(400, "request body must be a JSON object")
        return payload

    def _handle_save_graph(self, path: str, *, send_body: bool) -> None:
        """Persist the UI's ``qParams`` (pinned nodes, supernodes, etc.) back onto a graph's JSON file."""
        slug = path.removeprefix("/save_graph/").strip("/")
        if not slug or "/" in slug:
            raise RequestError(400, "missing or invalid graph slug")
        payload = self._read_json_body()
        qparams = payload.get("qParams")
        if not isinstance(qparams, dict):
            raise RequestError(400, "qParams must be an object")
        graph_path = safe_join(self.server.app.graph_file_dir, f"{slug}.json")
        if not graph_path.exists():
            raise RequestError(404, f"graph not found: {slug}")
        with graph_path.open(encoding="utf-8") as handle:
            graph = json.load(handle)
        graph["qParams"] = qparams
        write_json(graph_path, graph)
        self._send_json({"ok": True}, send_body=send_body)

    def _serve_metadata(self, *, send_body: bool) -> None:
        """Serve ``graph-metadata.json`` if it exists, else an empty graph list."""
        metadata_path = self.server.app.graph_file_dir / "graph-metadata.json"
        if metadata_path.exists():
            self._serve_file(
                self.server.app.graph_file_dir,
                "graph-metadata.json",
                send_body=send_body,
                no_store=True,
            )
            return
        self._send_json({"graphs": []}, send_body=send_body)

    def _serve_file(
        self, root: Path, rel_path: str, *, send_body: bool, no_store: bool = False
    ) -> None:
        """Serve a file under ``root``, gzip-compressing large JSON payloads."""
        path = safe_join(root, rel_path)
        if not path.exists() or not path.is_file():
            raise RequestError(404, "file not found")
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if len(content) > GZIP_MIN_BYTES and path.suffix == ".json":
            content = gzip.compress(content, compresslevel=3)
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
        else:
            self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            self.wfile.write(content)

    def _send_json(
        self, payload: dict[str, Any], *, status: int = 200, send_body: bool = True
    ) -> None:
        """Send ``payload`` as a JSON response with the given status code."""
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            self.wfile.write(content)


class BiologyTCPServer(ReusableThreadingTCPServer):
    """A threading TCP server that carries a reference to the shared :class:`BiologyApp` state."""

    def __init__(
        self, server_address: tuple[str, int], handler: type[BiologyRequestHandler], app: BiologyApp
    ):
        self.app = app
        super().__init__(server_address, handler)


class Server:
    """Handle for a running viewer server; stops it on demand or automatically at process exit."""

    def __init__(self, httpd: BiologyTCPServer, server_thread: threading.Thread) -> None:
        self.httpd = httpd
        self.server_thread = server_thread
        self._stopped = False
        atexit.register(self.stop)

    def stop(self) -> None:
        """Shut down the app and HTTP server; idempotent, and safe to call from ``atexit``."""
        if self._stopped:
            return
        self._stopped = True
        self.httpd.app.shutdown()
        self.httpd.shutdown()
        self.server_thread.join(timeout=5)
        self.httpd.server_close()
        with contextlib.suppress(ValueError):
            atexit.unregister(self.stop)


def serve(
    *,
    graph_file_dir: Path | str = DEFAULT_GRAPH_DIR,
    frontend_dir: Path | str | None = None,
    static_dir: Path | str = DEFAULT_STATIC_DIR,
    port: int = 8041,
    host: str = "",
) -> Server:
    """Start the viewer's HTTP server on a background thread and return a handle to it."""
    app = BiologyApp(
        graph_file_dir=graph_file_dir,
        frontend_dir=frontend_dir,
        static_dir=static_dir,
    )
    httpd = BiologyTCPServer((host, port), BiologyRequestHandler, app)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    logger.info("Serving biology graph viewer at http://localhost:%s", port)
    logger.info("Serving graph data from %s", app.graph_file_dir)
    logger.info("Serving circuit-tracer assets from %s", app.frontend_dir)
    return Server(httpd, server_thread)


def resolve_frontend_dir() -> Path:
    """Locate the circuit-tracer frontend assets directory next to this project's checkout."""
    adjacent = PROJECT_ROOT.parent / "circuit-tracer" / "circuit_tracer" / "frontend" / "assets"
    if adjacent.exists():
        return adjacent.resolve()
    raise RuntimeError(
        "Could not find circuit-tracer frontend assets at "
        f"{adjacent}. Pass --frontend-dir explicitly."
    )


def safe_join(root: Path, rel_path: str) -> Path:
    """Join and resolve ``rel_path`` under ``root``, raising 403 if it would escape ``root``."""
    root = root.resolve()
    path = (root / unquote(rel_path).lstrip("/")).resolve()
    if not path.is_relative_to(root):
        raise RequestError(403, "path escapes served directory")
    return path


def slugify(text: str) -> str:
    """Turn ``text`` into a filesystem-safe slug, falling back to a fixed default if empty."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip().lower()).strip("-")
    return slug or "uploaded-graph"


def upload_slug(
    graph: dict[str, Any],
    *,
    slug_override: str | None,
    filename: str | None,
) -> str:
    """Resolve the slug for an uploaded graph: explicit override, else metadata slug, else filename."""
    raw_slug = slug_override
    metadata = graph.get("metadata")
    if raw_slug is None and isinstance(metadata, dict):
        metadata_slug = metadata.get("slug")
        if isinstance(metadata_slug, str):
            raw_slug = metadata_slug
    if raw_slug is None and filename:
        raw_slug = Path(filename).stem

    slug = slugify(raw_slug or "uploaded-graph").strip(".")
    return slug or "uploaded-graph"


def normalize_uploaded_graph(
    raw_graph: dict[str, Any],
    slug: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate an uploaded graph's required fields and stamp its resolved ``slug`` onto metadata.

    Returns ``(graph, metadata)``. Raises :class:`RequestError` (400) if any
    required field (``metadata``, ``prompt_tokens``, ``nodes``, ``links``) is
    missing or malformed.
    """
    graph: dict[str, Any] = copy.deepcopy(raw_graph)

    metadata = graph.get("metadata")
    if not isinstance(metadata, dict):
        raise RequestError(400, "graph.metadata must be an object")

    prompt_tokens = metadata.get("prompt_tokens")
    if not isinstance(prompt_tokens, list) or not all(
        isinstance(token, str) for token in prompt_tokens
    ):
        raise RequestError(400, "graph.metadata.prompt_tokens must be a list of strings")

    if not isinstance(graph.get("nodes"), list):
        raise RequestError(400, "graph.nodes must be a list")
    if not isinstance(graph.get("links"), list):
        raise RequestError(400, "graph.links must be a list")

    qparams = graph.get("qParams", {})
    if not isinstance(qparams, dict):
        raise RequestError(400, "graph.qParams must be an object")

    normalized_metadata = dict(metadata)
    normalized_metadata["slug"] = slug
    graph["metadata"] = normalized_metadata
    graph["qParams"] = qparams
    return graph, normalized_metadata


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as pretty-printed, UTF-8 JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def optional_string(payload: dict[str, Any], key: str) -> str | None:
    """Return ``payload[key]`` stripped, or ``None`` if absent/blank; raises if present but not a string."""
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RequestError(400, f"{key} must be a string")
    value = value.strip()
    return value or None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the viewer server."""
    parser = argparse.ArgumentParser(description="Serve the llm-biology graph viewer.")
    parser.add_argument("--port", type=int, default=8041)
    parser.add_argument("--host", default="")
    parser.add_argument("--graph-file-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--frontend-dir", type=Path, default=None)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: start the viewer server and run until interrupted."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    server = serve(
        graph_file_dir=args.graph_file_dir,
        frontend_dir=args.frontend_dir,
        static_dir=args.static_dir,
        port=args.port,
        host=args.host,
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()
