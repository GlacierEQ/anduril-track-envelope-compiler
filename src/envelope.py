"""Track envelope compiler — bounded uncertainty from detections (sim)."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence


def digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Detection:
    sensor_id: str
    t: float
    x: float
    y: float
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence")


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
    residual_uncertainty: float
    fingerprint: str


class TrackEnvelopeCompiler:
    def __init__(self, pad: float = 1.0):
        self.pad = pad

    def compile(self, track_id: str, dets: Sequence[Detection]) -> TrackEnvelope | None:
        if not dets:
            return None
        xs = [d.x for d in dets]
        ys = [d.y for d in dets]
        ts = [d.t for d in dets]
        confs = [d.confidence for d in dets]
        mean_c = sum(confs) / len(confs)
        # residual: 1 - mean, inflated by spatial spread
        spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
        residual = min(1.0, (1.0 - mean_c) + math.tanh(spread / 100.0) * 0.2)
        body = {
            "id": track_id,
            "t": [min(ts), max(ts)],
            "x": [min(xs) - self.pad, max(xs) + self.pad],
            "y": [min(ys) - self.pad, max(ys) + self.pad],
            "mean_c": mean_c,
            "residual": residual,
            "n": len(dets),
        }
        return TrackEnvelope(
            track_id=track_id,
            t_min=min(ts),
            t_max=max(ts),
            x_min=min(xs) - self.pad,
            x_max=max(xs) + self.pad,
            y_min=min(ys) - self.pad,
            y_max=max(ys) + self.pad,
            mean_confidence=mean_c,
            residual_uncertainty=residual,
            fingerprint=digest(body),
        )
