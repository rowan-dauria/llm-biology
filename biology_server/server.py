"""HTTP server for prompt-preview and background attribution graph generation."""

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
import socketserver
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from biology_server.attribution import (
    DEFAULT_EDGE_TOP_K,
    DEFAULT_GRAPH_DIR,
    DEFAULT_LAYERS,
    DEFAULT_MAX_FEATURE_NODES,
    MODEL_ID,
    PROJECT_ROOT,
    BiologyAttributionRunner,
    GraphResult,
    PreviewResult,
    parse_layers,
    slugify,
)
from circuit_graph_export import write_graph_metadata

logger = logging.getLogger(__name__)
logger.propagate = False

DEFAULT_STATIC_DIR = Path(__file__).parent / "static"
GZIP_MIN_BYTES = 1 << 20
DEFAULT_MAX_JOBS = 200


@dataclass(slots=True)
class PreviewRecord:
    preview_id: str
    result: PreviewResult
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class GraphJob:
    job_id: str
    preview_id: str
    prompt: str
    slug: str
    target_token_id: int
    max_feature_nodes: int
    edge_top_k: int
    use_chat_template: bool
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    graph_path: str | None = None
    graph_url: str | None = None
    feature_nodes: int | None = None
    links: int | None = None


class BiologyApp:
    def __init__(
        self,
        *,
        graph_file_dir: Path | str = DEFAULT_GRAPH_DIR,
        frontend_dir: Path | str | None = None,
        static_dir: Path | str = DEFAULT_STATIC_DIR,
        runner: BiologyAttributionRunner | None = None,
        max_previews: int = 100,
        max_jobs: int = DEFAULT_MAX_JOBS,
    ) -> None:
        self.graph_file_dir = Path(graph_file_dir).resolve()
        self.frontend_dir = Path(frontend_dir).resolve() if frontend_dir else resolve_frontend_dir()
        self.static_dir = Path(static_dir).resolve()
        self.graph_file_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner or BiologyAttributionRunner(graph_file_dir=self.graph_file_dir)
        self.max_previews = max_previews
        self.max_jobs = max_jobs
        self.previews: dict[str, PreviewRecord] = {}
        self.jobs: dict[str, GraphJob] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="biology-graph")
        self._futures: dict[str, Future[None]] = {}

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = require_nonempty_string(payload, "prompt")
        slug = optional_string(payload, "slug")
        use_chat_template = optional_bool(payload, "use_chat_template", default=True)
        result = self.runner.preview(prompt, slug=slug, use_chat_template=use_chat_template)
        preview_id = uuid.uuid4().hex
        with self._lock:
            self.previews[preview_id] = PreviewRecord(preview_id=preview_id, result=result)
            self._trim_previews()
        return preview_to_json(preview_id, result)

    def enqueue_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
        preview_id = require_nonempty_string(payload, "preview_id")
        slug_override = optional_string(payload, "slug")
        max_feature_nodes = optional_int(
            payload,
            "max_feature_nodes",
            default=DEFAULT_MAX_FEATURE_NODES,
            minimum=1,
        )
        edge_top_k = optional_int(
            payload,
            "edge_top_k",
            default=DEFAULT_EDGE_TOP_K,
            minimum=0,
        )
        with self._lock:
            preview = self.previews.get(preview_id)
            if preview is None:
                raise RequestError(404, f"unknown preview_id: {preview_id}")
            preview_result = preview.result
            slug = slug_override or preview_result.slug
            job_id = uuid.uuid4().hex
            job = GraphJob(
                job_id=job_id,
                preview_id=preview_id,
                prompt=preview_result.prompt,
                slug=slug,
                target_token_id=preview_result.target_token_id,
                max_feature_nodes=max_feature_nodes,
                edge_top_k=edge_top_k,
                use_chat_template=preview_result.use_chat_template,
            )
            self.jobs[job_id] = job
            self._futures[job_id] = self._executor.submit(self._run_graph_job, job_id)
            self._trim_jobs()
        return job_to_json(job)

    def upload_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
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

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise RequestError(404, f"unknown job_id: {job_id}")
            return job_to_json(job)

    def _run_graph_job(self, job_id: str) -> None:
        with self._lock:
            job = self.jobs[job_id]
            job.status = "running"
            job.started_at = time.time()

        capture = StringIO()
        try:
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
                result = self.runner.generate_graph(
                    job.prompt,
                    slug=job.slug,
                    target_token_id=job.target_token_id,
                    max_feature_nodes=job.max_feature_nodes,
                    edge_top_k=job.edge_top_k,
                    graph_file_dir=self.graph_file_dir,
                    use_chat_template=job.use_chat_template,
                )
            self._finish_job(job_id, result, capture.getvalue())
        except Exception as exc:
            details = capture.getvalue() + traceback.format_exc()
            with self._lock:
                job = self.jobs[job_id]
                job.status = "failed"
                job.finished_at = time.time()
                job.error = str(exc)
                job.logs = split_logs(details)

    def _finish_job(self, job_id: str, result: GraphResult, logs: str) -> None:
        with self._lock:
            job = self.jobs[job_id]
            job.status = "succeeded"
            job.finished_at = time.time()
            job.graph_path = str(result.graph_path)
            job.graph_url = f"/graph_data/{result.slug}.json"
            job.feature_nodes = len(result.selected_features)
            job.links = len(result.links)
            job.logs = split_logs(logs)

    def _trim_previews(self) -> None:
        if len(self.previews) <= self.max_previews:
            return
        oldest = sorted(self.previews.values(), key=lambda item: item.created_at)
        for record in oldest[: len(self.previews) - self.max_previews]:
            self.previews.pop(record.preview_id, None)

    def _trim_jobs(self) -> None:
        if len(self.jobs) <= self.max_jobs:
            return
        finished = [job for job in self.jobs.values() if job.status in {"succeeded", "failed"}]
        finished.sort(key=lambda item: item.finished_at or item.created_at)
        for job in finished[: len(self.jobs) - self.max_jobs]:
            self.jobs.pop(job.job_id, None)
            self._futures.pop(job.job_id, None)


class RequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class ReusableThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class BiologyRequestHandler(http.server.SimpleHTTPRequestHandler):
    server: BiologyTCPServer

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        server = args[2]
        super().__init__(*args, directory=str(server.app.static_dir), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
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
        try:
            self._handle(send_body=send_body)
        except RequestError as exc:
            self._send_json({"error": exc.message}, status=exc.status, send_body=send_body)
        except Exception as exc:
            logger.exception("Error handling %s %s: %s", self.command, self.path, exc)
            self._send_json({"error": "internal server error"}, status=500, send_body=send_body)

    def _handle(self, *, send_body: bool) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if self.command == "POST" and path == "/api/preview":
            self._send_json(self.server.app.preview(self._read_json_body()))
            return
        if self.command == "POST" and path == "/api/graphs":
            self._send_json(self.server.app.enqueue_graph(self._read_json_body()), status=202)
            return
        if self.command == "POST" and path == "/api/upload_graph":
            self._send_json(self.server.app.upload_graph(self._read_json_body()), status=201)
            return
        if self.command == "GET" and path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/").strip("/")
            self._send_json(self.server.app.get_job(job_id), send_body=send_body)
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
                # Graph JSON is mutated in place by Save and the frontend fetches
                # it with cache:'force-cache'; no-store stops a reload from serving
                # a stale graph and dropping saved qParams. Feature files under
                # features/ are immutable, so leave them cacheable.
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
            # App shell changes during development; keep it uncached so an edit
            # is picked up on reload instead of serving a stale page/script.
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
        with graph_path.open("w", encoding="utf-8") as handle:
            json.dump(graph, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        self._send_json({"ok": True}, send_body=send_body)

    def _serve_metadata(self, *, send_body: bool) -> None:
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
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            self.wfile.write(content)


class BiologyTCPServer(ReusableThreadingTCPServer):
    def __init__(
        self, server_address: tuple[str, int], handler: type[BiologyRequestHandler], app: BiologyApp
    ):
        self.app = app
        super().__init__(server_address, handler)


class Server:
    def __init__(self, httpd: BiologyTCPServer, server_thread: threading.Thread) -> None:
        self.httpd = httpd
        self.server_thread = server_thread
        self._stopped = False
        atexit.register(self.stop)

    def stop(self) -> None:
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
    runner: BiologyAttributionRunner | None = None,
    port: int = 8041,
    host: str = "",
) -> Server:
    app = BiologyApp(
        graph_file_dir=graph_file_dir,
        frontend_dir=frontend_dir,
        static_dir=static_dir,
        runner=runner,
    )
    httpd = BiologyTCPServer((host, port), BiologyRequestHandler, app)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    logger.info("Serving biology backend at http://localhost:%s", port)
    logger.info("Serving graph data from %s", app.graph_file_dir)
    logger.info("Serving circuit-tracer assets from %s", app.frontend_dir)
    return Server(httpd, server_thread)


def resolve_frontend_dir() -> Path:
    adjacent = PROJECT_ROOT.parent / "circuit-tracer" / "circuit_tracer" / "frontend" / "assets"
    if adjacent.exists():
        return adjacent.resolve()
    raise RuntimeError(
        "Could not find circuit-tracer frontend assets at "
        f"{adjacent}. Pass --frontend-dir explicitly."
    )


def safe_join(root: Path, rel_path: str) -> Path:
    root = root.resolve()
    path = (root / unquote(rel_path).lstrip("/")).resolve()
    if not path.is_relative_to(root):
        raise RequestError(403, "path escapes served directory")
    return path


def upload_slug(
    graph: dict[str, Any],
    *,
    slug_override: str | None,
    filename: str | None,
) -> str:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def require_nonempty_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RequestError(400, f"{key} must be a non-empty string")
    return value


def optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RequestError(400, f"{key} must be a string")
    value = value.strip()
    return value or None


def optional_int(payload: dict[str, Any], key: str, *, default: int, minimum: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise RequestError(400, f"{key} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RequestError(400, f"{key} must be an integer") from exc
    if parsed < minimum:
        raise RequestError(400, f"{key} must be >= {minimum}")
    return parsed


def optional_bool(payload: dict[str, Any], key: str, *, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise RequestError(400, f"{key} must be a boolean")
    return value


def preview_to_json(preview_id: str, result: PreviewResult) -> dict[str, Any]:
    return {
        "preview_id": preview_id,
        "slug": result.slug,
        "prompt": result.prompt,
        "use_chat_template": result.use_chat_template,
        "prompt_tokens": result.prompt_tokens,
        "input_token_ids": result.input_token_ids,
        "target_token": {
            "id": result.target_token_id,
            "text": result.target_token_str,
            "prob": result.target_token_prob,
        },
        "top_tokens": [
            {"id": item.token_id, "text": item.token, "prob": item.prob}
            for item in result.top_tokens
        ],
    }


def job_to_json(job: GraphJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "preview_id": job.preview_id,
        "status": job.status,
        "slug": job.slug,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error,
        "logs": job.logs,
        "graph_path": job.graph_path,
        "graph_url": job.graph_url,
        "feature_nodes": job.feature_nodes,
        "links": job.links,
    }


def split_logs(logs: str) -> list[str]:
    return logs.splitlines()[-400:]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the llm-biology prompt backend.")
    parser.add_argument("--port", type=int, default=8041)
    parser.add_argument("--host", default="")
    parser.add_argument("--graph-file-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--frontend-dir", type=Path, default=None)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument("--layers", default=",".join(str(layer) for layer in DEFAULT_LAYERS))
    parser.add_argument("--model-id", default=MODEL_ID)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    runner = BiologyAttributionRunner(
        layers=parse_layers(args.layers),
        model_id=args.model_id,
        graph_file_dir=args.graph_file_dir,
    )
    server = serve(
        graph_file_dir=args.graph_file_dir,
        frontend_dir=args.frontend_dir,
        static_dir=args.static_dir,
        runner=runner,
        port=args.port,
        host=args.host,
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()
