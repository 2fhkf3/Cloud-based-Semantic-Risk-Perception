"""Bounded asynchronous cloud work; camera I/O and HTTP stay off the control path."""
from concurrent.futures import ThreadPoolExecutor
import base64
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler
import uuid
from .schema import SCHEMA_VERSION, STATE_NAMES, SemanticRisk, validate_request
from .session import SessionGuard

MAX_RESPONSE_BYTES = 64 * 1024


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None  # Never forward a bearer token to a redirected endpoint.


class RiskHTTPTransport:
    """A disposable process bounds DNS, file reads and slow HTTP responses alike."""
    def __init__(self, endpoint, token=None, timeout_s=3.0):
        parsed = urlparse(endpoint)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('endpoint must be an HTTP(S) URL without embedded credentials')
        if parsed.scheme == 'http' and parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('use HTTPS for a remote cloud endpoint; HTTP is loopback-only')
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('invalid HTTP timeout')
        self.endpoint, self.token, self.timeout_s = endpoint, token, timeout_s
        self.lock = threading.Lock()
        self.processes = set()
        self.closed = False

    def _call(self, payload, image_path=None, media_type=None):
        deadline = time.monotonic() + self.timeout_s
        job = dict(endpoint=self.endpoint, token=self.token, timeout_s=self.timeout_s,
                   payload=payload, image_path=image_path, media_type=media_type)
        raw_job = json.dumps(job, allow_nan=False).encode('utf-8')
        with self.lock:
            if self.closed:
                raise RuntimeError('cloud transport is closed')
            # A fresh interpreter avoids re-importing a caller's torch/GPU entry point.
            # Secrets and image contents are passed over stdin, not command-line arguments.
            process = subprocess.Popen(
                [sys.executable, '-m', 'navigation.http_worker'],
                cwd=str(Path(__file__).resolve().parents[1]),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
            )
            self.processes.add(process)
        try:
            try:
                stdout, _ = process.communicate(raw_job, timeout=max(0., deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.communicate()
                raise TimeoutError('cloud request exceeded its total deadline') from exc
            if time.monotonic() > deadline:
                raise TimeoutError('cloud request exceeded its total deadline')
            if process.returncode != 0 or not stdout:
                raise ConnectionError('cloud request worker failed or was cancelled')
            result = json.loads(stdout)
            if result.get('kind') == 'http_error':
                raise HTTPError(self.endpoint, result['code'], result['reason'], None, None)
            if result.get('kind') != 'ok':
                raise ConnectionError('cloud request failed: ' + str(result.get('kind', 'unknown')))
            return result['response']
        finally:
            with self.lock:
                self.processes.discard(process)

    def __call__(self, payload):
        return self._call(payload)

    def from_image_path(self, payload, image_path, media_type):
        return self._call(payload, str(image_path), media_type)

    def cancel_current(self):
        with self.lock:
            for process in tuple(self.processes):
                if process.poll() is None:
                    try:
                        process.kill()
                    except OSError:
                        pass

    def close(self):
        with self.lock:
            self.closed = True
        self.cancel_current()


class CloudRiskClient:
    def __init__(self, transport, interval_s=1.0, max_age_s=2.0):
        if not all(math.isfinite(v) and v > 0 for v in (interval_s, max_age_s)):
            raise ValueError('request interval and result TTL must be positive')
        self.transport, self.interval_s, self.max_age_s = transport, interval_s, max_age_s
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='semantic-risk')
        self.future = None
        self.request = None
        self.sessions = SessionGuard()
        self.last_submit = -math.inf
        self.latest = None
        self.last_error = None
        self.closed = False

    @property
    def session_id(self):
        return self.sessions.current

    def start_session(self, session_id):
        if self.sessions.start(session_id):
            self.latest = None
            self.last_submit = -math.inf
            if self.future is not None:
                self.future.cancel()
            if hasattr(self.transport, 'cancel_current'):
                self.transport.cancel_current()

    def reset(self, session_id):
        self.start_session(session_id)

    def poll(self, observation, now_s):
        if not math.isfinite(now_s):
            raise ValueError('now_s must be finite')
        if not self.sessions.accepts(observation.session_id):
            return None
        if self.future is not None and not self.future.done() and now_s - self.request['timestamp_s'] > self.max_age_s:
            self.last_error = 'expired_request'
            if hasattr(self.transport, 'cancel_current'):
                self.transport.cancel_current()
        if self.future is not None and self.future.done():
            try:
                result = SemanticRisk.from_response(self.future.result(), self.request)
                if result.session_id == self.session_id and 0 <= now_s - result.timestamp_s <= self.max_age_s:
                    self.latest = result
                    self.last_error = None
                else:
                    self.last_error = 'expired_or_previous_session'
            except Exception as exc:
                self.last_error = type(exc).__name__ + ': ' + str(exc)[:200]
            finally:
                self.future = self.request = None
        if self.latest is not None and not 0 <= now_s - self.latest.timestamp_s <= self.max_age_s:
            self.latest = None
        return self.latest

    def _request(self, observation, frame_timestamp_s, now_s):
        self.poll(observation, now_s)
        if (self.closed or observation.session_id != self.session_id or self.future is not None
                or now_s - self.last_submit < self.interval_s
                or not 0 <= now_s - observation.timestamp_s <= self.max_age_s):
            return None
        payload = {
            'schema_version': SCHEMA_VERSION, 'request_id': uuid.uuid4().hex,
            'session_id': observation.session_id, 'sequence': observation.sequence,
            'timestamp_s': float(observation.timestamp_s), 'frame_timestamp_s': float(frame_timestamp_s),
            'state_names': list(STATE_NAMES), 'state': observation.cloud_state(),
            'yaw_rad': float(observation.yaw_rad), 'task': 'risk_analysis',
        }
        validate_request(payload)
        return payload

    def submit(self, observation, image_bytes, frame_timestamp_s, now_s, media_type='image/png'):
        payload = self._request(observation, frame_timestamp_s, now_s)
        if payload is None:
            return False
        if not isinstance(image_bytes, bytes) or not 0 < len(image_bytes) <= 2 * 1024 * 1024:
            raise ValueError('image must contain 1..2MiB bytes')
        if media_type not in ('image/png', 'image/jpeg'):
            raise ValueError('only PNG/JPEG frames are supported')
        payload['image'] = {'media_type': media_type, 'base64': base64.b64encode(image_bytes).decode('ascii')}
        self.request = payload
        self.future = self.executor.submit(self.transport, payload)
        self.last_submit = now_s
        return True

    def submit_path(self, observation, image_path, frame_timestamp_s, now_s, media_type='image/png'):
        payload = self._request(observation, frame_timestamp_s, now_s)
        if payload is None:
            return False
        if media_type not in ('image/png', 'image/jpeg') or not hasattr(self.transport, 'from_image_path'):
            raise ValueError('path submission requires a PNG/JPEG-capable transport')
        self.request = payload
        self.future = self.executor.submit(self.transport.from_image_path, payload, str(image_path), media_type)
        self.last_submit = now_s
        return True

    def close(self):
        self.closed = True
        if hasattr(self.transport, 'close'):
            self.transport.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
