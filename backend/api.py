import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from embedding.query_rag import build_context, generate_answer, retrieve_chunks


HOST = "127.0.0.1"
PORT = 8000
MAX_REQUEST_SIZE = 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def source_strings(results: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    metadatas = results.get("metadatas", [[]])[0]

    for metadata in metadatas:
        raw_urls = metadata.get("source_urls", "")
        if isinstance(raw_urls, str):
            try:
                urls = json.loads(raw_urls)
            except json.JSONDecodeError:
                urls = [raw_urls] if raw_urls else []
        else:
            urls = raw_urls if isinstance(raw_urls, list) else []

        for source in urls:
            if isinstance(source, str) and source and source not in sources:
                sources.append(source)

    return sources


class RagRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/rag-query":
            json_response(self, 404, {"error": "Not found"})
            return

        content_length = self.headers.get("Content-Length")
        try:
            request_size = int(content_length) if content_length else 0
        except ValueError:
            json_response(self, 400, {"error": "Invalid Content-Length"})
            return

        if request_size > MAX_REQUEST_SIZE:
            json_response(self, 413, {"error": "Request body is too large"})
            return

        try:
            payload = json.loads(self.rfile.read(request_size))
        except (UnicodeDecodeError, json.JSONDecodeError):
            json_response(self, 400, {"error": "Request body must be valid JSON"})
            return

        query = payload.get("query") if isinstance(payload, dict) else None
        if not isinstance(query, str) or not query.strip():
            json_response(self, 400, {"error": "Query must be a non-empty string"})
            return

        try:
            results = retrieve_chunks(query.strip())
            answer = generate_answer(query.strip(), build_context(results))
            json_response(
                self,
                200,
                {"answer": str(answer), "sources": source_strings(results)},
            )
        except Exception as exc:
            logger.exception("RAG query failed: %s", exc)
            json_response(self, 500, {"error": "RAG query failed"})

    def do_GET(self):
        json_response(self, 405, {"error": "Method not allowed"})

    def log_message(self, format, *args):
        logger.info("%s - %s", self.address_string(), format % args)


def main():
    server = ThreadingHTTPServer((HOST, PORT), RagRequestHandler)
    logger.info("RAG API listening on http://%s:%s", HOST, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping RAG API")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
