"""Run with python -m inference.serve. Does not send any flight commands."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
from .service import MAX_REQUEST_BYTES, QwenBackend, RiskService


def make_server(service, host='127.0.0.1', port=8080, token=None):
    if host not in ('127.0.0.1', 'localhost', '::1') and not token:
        raise ValueError('non-loopback serving requires SEMANTIC_API_TOKEN and an HTTPS reverse proxy')
    busy = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5.0)

        def reply(self, status, data):
            raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == '/healthz':
                self.reply(200, {'status': 'ready', 'backend': type(service.backend).__name__, 'schema_version': 1})
            else:
                self.reply(404, {'status': 'not_found'})

        def do_POST(self):
            if self.path != '/v1/semantic-risk':
                self.reply(404, {'status': 'not_found'})
                return
            if token and not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                self.reply(401, {'status': 'unauthorized'})
                return
            if self.headers.get_content_type() != 'application/json' or self.headers.get('Transfer-Encoding'):
                self.reply(400, {'status': 'invalid_content_type_or_encoding'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                size = 0
            if not 0 < size <= MAX_REQUEST_BYTES:
                self.reply(413, {'status': 'invalid_body_size'})
                return
            if not busy.acquire(blocking=False):
                self.reply(503, {'status': 'busy'})
                return
            try:
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError('incomplete request body')
                self.reply(200, service.analyze(json.loads(body)))
            except (ValueError, TypeError, KeyError, OSError):
                self.reply(422, {'status': 'invalid_request_or_model_result'})
            except Exception:
                # Do not leak local model paths, credentials or server tracebacks to callers.
                self.reply(500, {'status': 'inference_error'})
            finally:
                busy.release()

        def log_message(self, fmt, *args):
            pass  # No raw frames, prompts or tokens in automatic request logs.

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8080, type=int)
    parser.add_argument('--max-new-tokens', default=256, type=int)
    args = parser.parse_args()
    token = os.getenv('SEMANTIC_API_TOKEN')
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not token:
        parser.error('set SEMANTIC_API_TOKEN before exposing the service')
    service = RiskService(QwenBackend(args.model, args.adapter, args.max_new_tokens))
    server = make_server(service, args.host, args.port, token)
    print(f'Cloud risk API ready on {args.host}:{args.port}; model loaded; no flight actuation.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
