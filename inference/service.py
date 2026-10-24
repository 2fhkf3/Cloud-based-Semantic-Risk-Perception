"""Online adapter around the SAME Qwen/LoRA JSONL inference implementation."""
from __future__ import annotations
import base64
import binascii
import io
import json
import tempfile
from pathlib import Path
from navigation.schema import validate_request, SemanticRisk
from .build_state_inference import state_text

MAX_REQUEST_BYTES = 3 * 1024 * 1024


def risk_prompt(payload):
    return (
        f"{state_text(payload['state'])}。yaw_rad={payload['yaw_rad']:g}。\n"
        '请基于图像与状态分析无人机导航的语义风险，不要臆测不可见障碍。'
        '仅返回一个JSON对象，包含risk_level（low/medium/high/critical）、'
        'risk_score（0到1）、confidence（0到1）及reason（原因文字）。'
        '无法判断时confidence取0并解释原因；不得输出飞行指令或电机控制量。'
        'risk_score是风险指标而不是经过校准的碰撞概率。'
    )


def parse_risk(answer):
    if not isinstance(answer, str) or len(answer) > 32768:
        raise ValueError('invalid or oversized model answer')
    text = answer.strip()
    if text.startswith('```') and text.endswith('```'):
        lines = text.splitlines()
        if lines[0] not in ('```', '```json'):
            raise ValueError('unsupported code fence')
        text = '\n'.join(lines[1:-1])
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError('model must return one JSON object')
    # No invented confidence and no regex-based salvage from arbitrary prose.
    return result


class QwenBackend:
    def __init__(self, model, adapter=None, max_new_tokens=256):
        # Heavy dependencies are optional for protocol tests and local-only navigation.
        from .infer_jsonl import get_device, predict
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from peft import PeftModel
        self.device, dtype = get_device()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, torch_dtype=dtype)
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, str(adapter))
        self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model)
        self.predict = predict
        self.max_new_tokens = max_new_tokens

    def __call__(self, image, prompt):
        # Per-request isolated scratch directory, automatically removed; never accept client file paths.
        with tempfile.TemporaryDirectory(prefix='semantic-frame-') as directory:
            path = Path(directory) / 'frame.png'
            path.write_bytes(image)
            return self.predict(self.model, self.processor, self.device,
                                {'image': str(path), 'prompt': prompt}, self.max_new_tokens)


class RiskService:
    def __init__(self, backend):
        self.backend = backend

    def analyze(self, payload):
        validate_request(payload)
        image = payload.get('image')
        if not isinstance(image, dict) or image.get('media_type') not in ('image/png', 'image/jpeg'):
            raise ValueError('image must specify PNG/JPEG content')
        encoded = image.get('base64')
        if not isinstance(encoded, str) or len(encoded) > 2800000:
            raise ValueError('missing or oversized base64 image')
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError('invalid base64 image') from exc
        if not 0 < len(raw) <= 2 * 1024 * 1024:
            raise ValueError('image too large or empty')
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as decoded:
            if decoded.width * decoded.height > 4096 * 4096 or decoded.format not in ('PNG', 'JPEG'):
                raise ValueError('unsupported image size/format')
            if ('image/png' if decoded.format == 'PNG' else 'image/jpeg') != image['media_type']:
                raise ValueError('declared image type does not match actual image')
            decoded.verify()
        result = parse_risk(self.backend(raw, risk_prompt(payload)))
        response = {k: payload[k] for k in ('schema_version', 'request_id', 'session_id', 'sequence', 'timestamp_s')}
        response.update(status='ok', risk=result)
        SemanticRisk.from_response(response, payload)  # Reject malformed model output server-side too.
        return response
