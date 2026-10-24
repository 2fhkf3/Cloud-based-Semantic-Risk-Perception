"""RTV -> EWAF -> CVaR, matching the existing GAP8 reference policy."""
from dataclasses import dataclass
import math


@dataclass
class ART:
    w0: float = -3.0
    w1: float = 3.0
    previous_rtv: float | None = None

    @property
    def alpha(self) -> float:
        m = max(self.w0, self.w1)
        a, b = math.exp(self.w0 - m), math.exp(self.w1 - m)
        total = a + b
        return (b + total / 9.0) / (total + total / 9.0)

    def accept(self, rtv: float):
        if not math.isfinite(rtv) or rtv < 0:
            raise ValueError('RTV must be finite and nonnegative')
        if self.previous_rtv is not None:
            delta = 0.5 * (rtv - self.previous_rtv)
            self.w0 += delta
            self.w1 -= delta
        self.previous_rtv = rtv


@dataclass(frozen=True)
class FusionConfig:
    strength: float = 0.25
    min_confidence: float = 0.6
    max_age_s: float = 2.0
    alpha_min: float = 0.1

    def __post_init__(self):
        if not (0 <= self.strength <= 1 and 0 <= self.min_confidence <= 1
                and 0 < self.alpha_min <= 1 and math.isfinite(self.max_age_s) and self.max_age_s > 0):
            raise ValueError('invalid semantic fusion configuration')

    def apply(self, alpha, risk, observation, now_s):
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError('ART alpha must be finite and in (0,1]')
        if risk is None:
            return alpha, 'local_only'
        if risk.session_id != observation.session_id or risk.sequence > observation.sequence:
            return alpha, 'wrong_session_or_future_sequence'
        age = now_s - risk.timestamp_s
        if not 0 <= age <= self.max_age_s:
            return alpha, 'expired_or_future_risk'
        if risk.confidence < self.min_confidence:
            return alpha, 'low_confidence'
        # Cloud never raises alpha or modifies the learned Q values directly.
        return min(alpha, max(self.alpha_min, alpha - self.strength * risk.confidence * risk.risk_score)), 'semantic_fusion'
