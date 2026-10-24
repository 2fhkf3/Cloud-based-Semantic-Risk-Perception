"""Disposable deadline-isolated file/HTTP worker; internal interface, no flight code."""
import base64
import json
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, build_opener
from .cloud import MAX_RESPONSE_BYTES, NoRedirect


def main():
    try:
        job = json.load(sys.stdin)
        payload = job['payload']
        if job.get('image_path') is not None:
            with Path(job['image_path']).open('rb') as handle:
                image = handle.read(2 * 1024 * 1024 + 1)
            if not 0 < len(image) <= 2 * 1024 * 1024:
                raise ValueError('empty or oversized image')
            payload['image'] = dict(media_type=job['media_type'], base64=base64.b64encode(image).decode('ascii'))
        headers = {'Content-Type': 'application/json'}
        if job.get('token'):
            headers['Authorization'] = 'Bearer ' + job['token']
        request = Request(job['endpoint'], json.dumps(payload, allow_nan=False).encode('utf-8'), headers, method='POST')
        with build_opener(NoRedirect()).open(request, timeout=job['timeout_s']) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError('oversized cloud response')
        result = dict(kind='ok', response=json.loads(raw))
    except HTTPError as exc:
        result = dict(kind='http_error', code=exc.code, reason=str(exc.reason))
    except Exception as exc:
        result = dict(kind=type(exc).__name__)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
