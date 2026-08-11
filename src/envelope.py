"""Track envelope compiler — uncertainty-preserving multi-sensor envelopes."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence


def digest(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(name)


@dataclass(frozen=True)
class Detection:
    sensor_id: str
    t: float
    x: float
    y: float
    confidence: float
    var_x: float = 0.0
    var_y: float = 0.0
    cov_xy: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_id, str) or not self.sensor_id.strip():
            raise ValueError("sensor_id")
        for name in ("t", "x", "y", "confidence", "var_x", "var_y", "cov_xy"):
            _finite(name, float(getattr(self, name)))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence")
        if self.var_x < 0.0 or self.var_y < 0.0:
            raise ValueError("variance")
        if self.var_x * self.var_y - self.cov_xy * self.cov_xy < -1e-12:
            raise ValueError("covariance_not_psd")


@dataclass(frozen=True)
class TrackEnvelope:
    track_id: str
    t_min: float
    t_max: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    mean_confidence: float
    fused_confidence: float
    residual_unknown_mass: float
    residual_uncertainty: float
    sensor_count: int
    reference_t: float
    policy_fingerprint: str
    fingerprint: str


class TrackEnvelopeCompiler:
    """Compile detections without hiding age, covariance, or unknown mass.

    Repeated observations from one sensor may change geometry, but they do not count
    as independent confidence. Per-sensor support is the strongest temporally decayed
    observation, then independent sensor supports are fused while retaining the
    unfused probability mass as ``residual_unknown_mass``.
    """

    def __init__(
        self,
        pad: float = 1.0,
        half_life: float = 5.0,
        covariance_sigma: float = 2.0,
    ) -> None:
        for name, value in (
            ("pad", pad),
            ("half_life", half_life),
            ("covariance_sigma", covariance_sigma),
        ):
            _finite(name, float(value))
        if pad < 0.0:
            raise ValueError("pad")
        if half_life <= 0.0:
            raise ValueError("half_life")
        if covariance_sigma < 0.0:
            raise ValueError("covariance_sigma")
        self.pad = float(pad)
        self.half_life = float(half_life)
        self.covariance_sigma = float(covariance_sigma)

    def compile(self, track_id: str, dets: Sequence[Detection]) -> TrackEnvelope | None:
        if not isinstance(track_id, str) or not track_id.strip():
            raise ValueError("track_id")
        if not dets:
            return None
        if any(not isinstance(d, Detection) for d in dets):
            raise TypeError("detections")

        reference_t = max(d.t for d in dets)
        sensor_support: dict[str, float] = {}
        x_lows: list[float] = []
        x_highs: list[float] = []
        y_lows: list[float] = []
        y_highs: list[float] = []
        sigmas: list[float] = []
        confs: list[float] = []

        for d in dets:
            age = reference_t - d.t
            decay = math.pow(0.5, age / self.half_life)
            support = d.confidence * decay
            sensor_support[d.sensor_id] = max(sensor_support.get(d.sensor_id, 0.0), support)

            sigma_x = math.sqrt(d.var_x)
            sigma_y = math.sqrt(d.var_y)
            x_growth = self.pad + self.covariance_sigma * sigma_x
            y_growth = self.pad + self.covariance_sigma * sigma_y
            x_lows.append(d.x - x_growth)
            x_highs.append(d.x + x_growth)
            y_lows.append(d.y - y_growth)
            y_highs.append(d.y + y_growth)
            sigmas.append(math.sqrt(d.var_x + d.var_y))
            confs.append(d.confidence)

        residual_unknown = 1.0
        for sensor_id in sorted(sensor_support):
            residual_unknown *= 1.0 - sensor_support[sensor_id]
        residual_unknown = min(1.0, max(0.0, residual_unknown))
        fused_confidence = 1.0 - residual_unknown

        raw_xs = [d.x for d in dets]
        raw_ys = [d.y for d in dets]
        spread = (max(raw_xs) - min(raw_xs)) + (max(raw_ys) - min(raw_ys))
        spatial_penalty = math.tanh(spread / 100.0) * 0.15
        mean_sigma = sum(sigmas) / len(sigmas)
        covariance_penalty = math.tanh(mean_sigma / 10.0) * 0.15
        residual = min(1.0, residual_unknown + spatial_penalty + covariance_penalty)
        mean_confidence = sum(confs) / len(confs)

        policy = {
            "pad": self.pad,
            "half_life": self.half_life,
            "covariance_sigma": self.covariance_sigma,
            "sensor_fusion": "max_temporally_decayed_support_per_sensor_then_unknown_product",
            "residual_rule": "unknown_mass_plus_spatial_and_covariance_penalties",
        }
        policy_fingerprint = digest(policy)
        evidence = [
            {
                "sensor_id": d.sensor_id,
                "t": d.t,
                "x": d.x,
                "y": d.y,
                "confidence": d.confidence,
                "var_x": d.var_x,
                "var_y": d.var_y,
                "cov_xy": d.cov_xy,
            }
            for d in sorted(
                dets,
                key=lambda item: (
                    item.sensor_id,
                    item.t,
                    item.x,
                    item.y,
                    item.confidence,
                    item.var_x,
                    item.var_y,
                    item.cov_xy,
                ),
            )
        ]
        body = {
            "track_id": track_id,
            "reference_t": reference_t,
            "evidence": evidence,
            "policy_fingerprint": policy_fingerprint,
            "bounds": [min(x_lows), max(x_highs), min(y_lows), max(y_highs)],
            "mean_confidence": mean_confidence,
            "fused_confidence": fused_confidence,
            "residual_unknown_mass": residual_unknown,
            "residual_uncertainty": residual,
            "sensor_count": len(sensor_support),
        }
        return TrackEnvelope(
            track_id=track_id,
            t_min=min(d.t for d in dets),
            t_max=reference_t,
            x_min=min(x_lows),
            x_max=max(x_highs),
            y_min=min(y_lows),
            y_max=max(y_highs),
            mean_confidence=mean_confidence,
            fused_confidence=fused_confidence,
            residual_unknown_mass=residual_unknown,
            residual_uncertainty=residual,
            sensor_count=len(sensor_support),
            reference_t=reference_t,
            policy_fingerprint=policy_fingerprint,
            fingerprint=digest(body),
        )
