"""Shared, versioned interface. Seven cloud values are NOT the GRU frame."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

STATE_NAMES = ('x', 'y', 'd_goal', 'd_front', 'd_left', 'd_back', 'd_right')
SCHEMA_VERSION = 1


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ValueError(f'{name} must be a nonempty string of at most 128 characters')
    return value


def sequence(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('sequence must be a positive integer')
    return value


@dataclass(frozen=True)
class Observation:
    session_id: str
    sequence: int
    timestamp_s: float  # Local monotonic/session clock; never compare to server wall time.
    position_m: tuple[float, float]
    goal_m: tuple[float, float]
    velocity_mps: tuple[float, float]
    ranges_m: tuple[float, float, float, float]  # body front,left,back,right
    yaw_rad: float = 0.0

    def __post_init__(self):
        identifier(self.session_id, 'session_id')
        sequence(self.sequence)
        finite(self.timestamp_s, 'timestamp_s')
        finite(self.yaw_rad, 'yaw_rad')
        for name, length in [('position_m', 2), ('goal_m', 2), ('velocity_mps', 2), ('ranges_m', 4)]:
            values = getattr(self, name)
            if len(values) != length:
                raise ValueError(f'{name} must contain {length} numbers')
            for value in values:
                finite(value, name)
        if any(not 0 <= value <= 4 for value in self.ranges_m):
            raise ValueError('ranges_m must be valid, clipped metric ranges in [0,4]')

    def cloud_state(self) -> list[float]:
        distance = math.hypot(*(g - p for g, p in zip(self.goal_m, self.position_m)))
        return [*self.position_m, distance, *self.ranges_m]

    def frame(self, previous_action: int | None) -> list[float]:
        if previous_action is not None and (type(previous_action) is not int or not 0 <= previous_action < 12):
            raise ValueError('previous_action must be an accepted action id in [0,11] or None')
        actions = [0.0] * 12
        if previous_action is not None:
            actions[previous_action] = 1.0
        return [*(g - p for g, p in zip(self.goal_m, self.position_m)),
                *self.velocity_mps, *self.ranges_m, *actions]

    @classmethod
    def from_dict(cls, data: dict) -> 'Observation':
        return cls(**{k: tuple(v) if k.endswith(('_m', '_mps')) and isinstance(v, list) else v
                      for k, v in data.items()})


def validate_request(data: dict) -> dict:
    if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data['schema_version'] != SCHEMA_VERSION:
        raise ValueError('unsupported schema_version')
    identifier(data.get('request_id'), 'request_id')
    identifier(data.get('session_id'), 'session_id')
    sequence(data.get('sequence'))
    timestamp = finite(data.get('timestamp_s'), 'timestamp_s')
    frame_time = finite(data.get('frame_timestamp_s'), 'frame_timestamp_s')
    if abs(timestamp - frame_time) > 0.1 + 1e-9:
        raise ValueError('image/state timestamps differ by more than 100 ms')
    state = data.get('state')
    if not isinstance(state, list) or len(state) != 7 or data.get('state_names') != list(STATE_NAMES):
        raise ValueError('state must follow the seven-value STATE_NAMES order')
    for value in state:
        finite(value, 'state')
    if state[2] < 0 or any(not 0 <= v <= 4 for v in state[3:]):
        raise ValueError('negative goal distance or out-of-range sensor value')
    finite(data.get('yaw_rad'), 'yaw_rad')
    if data.get('task') != 'risk_analysis':
        raise ValueError('online control assistance only consumes risk_analysis')
    return data


@dataclass(frozen=True)
class SemanticRisk:
    request_id: str
    session_id: str
    sequence: int
    timestamp_s: float
    risk_score: float
    confidence: float
    risk_level: str
    reason: str

    def __post_init__(self):
        identifier(self.request_id, 'request_id')
        identifier(self.session_id, 'session_id')
        sequence(self.sequence)
        finite(self.timestamp_s, 'timestamp_s')
        for name in ('risk_score', 'confidence'):
            if not 0 <= finite(getattr(self, name), name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        if self.risk_level not in ('low', 'medium', 'high', 'critical'):
            raise ValueError('invalid risk_level')
        if not isinstance(self.reason, str) or len(self.reason) > 4096:
            raise ValueError('invalid risk reason')

    @classmethod
    def from_response(cls, data: dict, request: dict) -> 'SemanticRisk':
        for key in ('schema_version', 'request_id', 'session_id', 'sequence', 'timestamp_s'):
            # bool/int equivalence must not let malformed IDs pass.
            if type(data.get(key)) is not type(request[key]) or data[key] != request[key]:
                raise ValueError(f'response does not match request {key}')
        if data.get('status') != 'ok':
            raise ValueError('cloud response is not successful')
        result = data.get('risk')
        if not isinstance(result, dict):
            raise ValueError('missing structured risk result')
        score = finite(result.get('risk_score'), 'risk_score')
        confidence = finite(result.get('confidence'), 'confidence')
        if not 0 <= score <= 1 or not 0 <= confidence <= 1:
            raise ValueError('risk_score/confidence must be in [0,1]')
        if result.get('risk_level') not in ('low', 'medium', 'high', 'critical'):
            raise ValueError('invalid risk_level')
        reason = result.get('reason')
        if not isinstance(reason, str) or len(reason) > 4096:
            raise ValueError('reason must be a bounded string')
        return cls(request['request_id'], request['session_id'], request['sequence'],
                   request['timestamp_s'], score, confidence, result['risk_level'], reason)
