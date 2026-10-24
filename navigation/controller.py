"""Commit valid observation windows transactionally; accepted actions alone update ART."""
from collections import deque
from dataclasses import asdict, dataclass
import math
import time
from .art import ART, FusionConfig
from .schema import Observation
from .session import SessionGuard

SPEEDS = tuple((math.exp((i + 1) / 3) - 1) / (math.e - 1) for i in range(3))
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def action_velocity(action):
    if type(action) is not int or not 0 <= action < 12:
        raise ValueError('invalid action')
    return tuple(v * SPEEDS[action % 3] for v in DIRECTIONS[action // 3])


@dataclass(frozen=True)
class Decision:
    session_id: str
    sequence: int
    action_id: int | None
    velocity_world_mps: tuple[float, float]
    art_alpha: float
    fused_alpha: float
    fusion_status: str
    rtv: float | None
    q_values: list[float]
    status: str = 'proposed'

    def to_dict(self):
        return asdict(self)


class Controller:
    def __init__(self, predictor, fusion=FusionConfig(), max_observation_age_s=0.2, yaw_tolerance_rad=0.1):
        if not math.isfinite(max_observation_age_s) or max_observation_age_s <= 0:
            raise ValueError('invalid observation deadline')
        if not math.isfinite(yaw_tolerance_rad) or not 0 <= yaw_tolerance_rad <= math.pi:
            raise ValueError('invalid yaw tolerance')
        self.predictor = predictor
        self.fusion = fusion
        self.max_age = max_observation_age_s
        self.yaw_tolerance = yaw_tolerance_rad
        self.sessions = SessionGuard()
        self.reset()

    @property
    def session_id(self):
        return self.sessions.current

    def start_session(self, session_id):
        if self.sessions.start(session_id):
            self.reset()

    def reset(self, session_id=None):
        if session_id is not None:
            self.sessions.start(session_id)
        self.last_sequence = 0
        self.last_time = None
        self.history = deque(maxlen=4)
        self.previous_action = None
        self.art = ART()
        self.pending = None
        self.goal_m = None

    def hold(self, obs, reason):
        previous_sequence, previous_time, previous_goal = self.last_sequence, self.last_time, self.goal_m
        self.reset()
        self.last_sequence, self.last_time, self.goal_m = previous_sequence, previous_time, previous_goal
        return Decision(obs.session_id, obs.sequence, None, (0., 0.), self.art.alpha,
                        self.art.alpha, reason, None, [], 'hold')

    def step(self, obs: Observation, now_s: float, semantic_risk=None, *, previous_executed_action='internal'):
        started = time.monotonic()
        if not math.isfinite(now_s):
            raise ValueError('now_s must be finite')
        if not self.sessions.accepts(obs.session_id):
            return Decision(obs.session_id, obs.sequence, None, (0., 0.), self.art.alpha,
                            self.art.alpha, 'inactive_session', None, [], 'rejected')
        if self.pending is not None:
            raise RuntimeError('acknowledge or reject the previous command before advancing')
        if self.goal_m is not None and tuple(obs.goal_m) != self.goal_m:
            return self.hold(obs, 'goal_change_requires_new_session')
        if not 0 <= now_s - obs.timestamp_s <= self.max_age:
            return self.hold(obs, 'stale_or_future_observation')
        yaw = math.atan2(math.sin(obs.yaw_rad), math.cos(obs.yaw_rad))
        if abs(yaw) > self.yaw_tolerance:
            return self.hold(obs, 'unsupported_yaw')
        if obs.sequence <= self.last_sequence or (self.last_time is not None and obs.timestamp_s <= self.last_time):
            return self.hold(obs, 'out_of_order_observation')
        previous_action = self.previous_action
        if previous_executed_action != 'internal':
            obs.frame(previous_executed_action)  # Validate without committing partially advanced state.
            previous_action = previous_executed_action
        frame = obs.frame(previous_action if self.history else None)
        candidate_history = deque((row[:] for row in self.history), maxlen=4)
        if not candidate_history:
            candidate_history.extend([frame[:] for _ in range(4)])
        else:
            candidate_history.append(frame)
        alpha = self.art.alpha
        fused, status = self.fusion.apply(alpha, semantic_risk, obs, now_s)
        action, q, rtv = self.predictor(list(candidate_history), fused)
        if type(action) is not int or not 0 <= action < 12 or len(q) != 12:
            raise ValueError('invalid model output')
        if not all(math.isfinite(v) for v in [*q, rtv]) or rtv < 0:
            raise ValueError('nonfinite or invalid model output')
        finished_s = now_s + (time.monotonic() - started)
        if not 0 <= finished_s - obs.timestamp_s <= self.max_age:
            return self.hold(obs, 'deadline_exceeded')
        self.history = candidate_history
        self.previous_action = previous_action
        self.last_sequence, self.last_time = obs.sequence, obs.timestamp_s
        self.goal_m = tuple(obs.goal_m)
        self.pending = Decision(obs.session_id, obs.sequence, action, action_velocity(action),
                                alpha, fused, status, rtv, q)
        return self.pending

    def acknowledge(self, decision: Decision, accepted: bool):
        if type(accepted) is not bool:
            raise ValueError('accepted must be an explicit boolean')
        if self.pending is not decision:
            raise ValueError('only the current pending decision can be acknowledged')
        if accepted:
            self.previous_action = decision.action_id
            self.art.accept(decision.rtv)
        else:
            # Reject means an actual zero-velocity hold, not execution of another discrete action.
            self.previous_action = None
        self.pending = None
