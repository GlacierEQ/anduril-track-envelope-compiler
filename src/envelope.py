"""Track envelope compiler — time-decayed, covariance-aware bounded uncertainty."""
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


@dataclass(frozen=True)
class Detection:
    sensor_id: str
    t: float
    x: float
    y: float
    confidence: float
    covariance_xx: float = 0.0
    covariance_yy: float = 0.0
    covariance_xy: float = 0.0

    def __post_init__(self) -> None:
        if not self.sensor_id:
            raise ValueError("sensor_id")
        if not all(
            math.isfinite(value)
            for value in (
                self.t,
                self.x,
                self.y,
                self.confidence,
                self.covariance_xx,
                self.covariance_yy,
                self.covariance_xy,
            )
        ):
            raise ValueError("finite detection values required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence")
        if self.covariance_xx < 0.0 or self.covariance_yy < 0.0:
            raise ValueError("covariance diagonal")
        determinant = (
            self.covariance_xx * self.covariance_yy
            - self.covariance_xy * self.covariance_xy
        )
        if determinant < -1e-12:
            raise ValueError("covariance must be positive semidefinite")


@dataclass(frozen=True)
class SensorFusion:
    sensor_id: str
    support: float
    fused_x: float
    fused_y: float
    freshest_t: float
    detections: int


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
    fused_x: float = 0.0
    fused_y: float = 0.0
    effective_confidence: float = 0.0
    residual_unknown_mass: float = 1.0
    sensor_count: int = 0
    reference_time: float = 0.0
    sensor_fusions: tuple[SensorFusion, ...] = ()


class TrackEnvelopeCompiler:
    def __init__(
        self,
        pad: float = 1.0,
        *,
        half_life: float = 10.0,
        sigma_scale: float = 2.0,
        age_covariance_growth: float = 0.25,
    ):
        if not math.isfinite(pad) or pad < 0.0:
            raise ValueError("pad")
        if not math.isfinite(half_life) or half_life <= 0.0:
            raise ValueError("half_life")
        if not math.isfinite(sigma_scale) or sigma_scale < 0.0:
            raise ValueError("sigma_scale")
        if not math.isfinite(age_covariance_growth) or age_covariance_growth < 0.0:
            raise ValueError("age_covariance_growth")
        self.pad = pad
        self.half_life = half_life
        self.sigma_scale = sigma_scale
        self.age_covariance_growth = age_covariance_growth

    def _decay(self, age: float) -> float:
        return math.exp(-math.log(2.0) * age / self.half_life)

    def compile(
        self,
        track_id: str,
        dets: Sequence[Detection],
        *,
        reference_time: float | None = None,
    ) -> TrackEnvelope | None:
        if not track_id:
            raise ValueError("track_id")
        if not dets:
            return None
        if reference_time is None:
            reference_time = max(d.t for d in dets)
        if not math.isfinite(reference_time):
            raise ValueError("reference_time")
        if any(d.t > reference_time + 1e-12 for d in dets):
            raise ValueError("reference_time precedes detection")

        by_sensor: dict[str, list[tuple[Detection, float]]] = {}
        expanded_x: list[tuple[float, float]] = []
        expanded_y: list[tuple[float, float]] = []
        confs = [d.confidence for d in dets]
        ts = [d.t for d in dets]

        for detection in dets:
            age = reference_time - detection.t
            temporal = self._decay(age)
            support = detection.confidence * temporal
            by_sensor.setdefault(detection.sensor_id, []).append((detection, support))

            age_growth = 1.0 + self.age_covariance_growth * age / self.half_life
            sigma_x = math.sqrt(detection.covariance_xx) * age_growth
            sigma_y = math.sqrt(detection.covariance_yy) * age_growth
            expanded_x.append(
                (
                    detection.x - self.pad - self.sigma_scale * sigma_x,
                    detection.x + self.pad + self.sigma_scale * sigma_x,
                )
            )
            expanded_y.append(
                (
                    detection.y - self.pad - self.sigma_scale * sigma_y,
                    detection.y + self.pad + self.sigma_scale * sigma_y,
                )
            )

        sensor_fusions: list[SensorFusion] = []
        residual_unknown_mass = 1.0
        fusion_weight_sum = 0.0
        fusion_x_sum = 0.0
        fusion_y_sum = 0.0
        for sensor_id in sorted(by_sensor):
            rows = by_sensor[sensor_id]
            total = sum(weight for _, weight in rows)
            if total > 0.0:
                sensor_x = sum(d.x * weight for d, weight in rows) / total
                sensor_y = sum(d.y * weight for d, weight in rows) / total
            else:
                sensor_x = sum(d.x for d, _ in rows) / len(rows)
                sensor_y = sum(d.y for d, _ in rows) / len(rows)
            # Repeated reports from one sensor can refine its centroid but cannot
            # manufacture more than one sensor's independent known-mass support.
            sensor_support = max(weight for _, weight in rows)
            residual_unknown_mass *= 1.0 - sensor_support
            fusion_weight_sum += sensor_support
            fusion_x_sum += sensor_x * sensor_support
            fusion_y_sum += sensor_y * sensor_support
            sensor_fusions.append(
                SensorFusion(
                    sensor_id=sensor_id,
                    support=sensor_support,
                    fused_x=sensor_x,
                    fused_y=sensor_y,
                    freshest_t=max(d.t for d, _ in rows),
                    detections=len(rows),
                )
            )

        if fusion_weight_sum > 0.0:
            fused_x = fusion_x_sum / fusion_weight_sum
            fused_y = fusion_y_sum / fusion_weight_sum
        else:
            fused_x = sum(d.x for d in dets) / len(dets)
            fused_y = sum(d.y for d in dets) / len(dets)

        x_min = min(low for low, _ in expanded_x)
        x_max = max(high for _, high in expanded_x)
        y_min = min(low for low, _ in expanded_y)
        y_max = max(high for _, high in expanded_y)
        mean_confidence = sum(confs) / len(confs)
        effective_confidence = 1.0 - residual_unknown_mass

        spatial_spread = (x_max - x_min) + (y_max - y_min)
        avg_sigma = sum(
            math.sqrt(d.covariance_xx) + math.sqrt(d.covariance_yy) for d in dets
        ) / (2.0 * len(dets))
        residual = min(
            1.0,
            residual_unknown_mass
            + math.tanh(spatial_spread / 100.0) * 0.1
            + math.tanh(avg_sigma / 10.0) * 0.1,
        )

        body = {
            "id": track_id,
            "t": [min(ts), max(ts)],
            "reference_time": reference_time,
            "x": [x_min, x_max],
            "y": [y_min, y_max],
            "mean_c": mean_confidence,
            "effective_c": effective_confidence,
            "residual_unknown_mass": residual_unknown_mass,
            "residual": residual,
            "sensors": [
                {
                    "id": row.sensor_id,
                    "support": row.support,
                    "fused": [row.fused_x, row.fused_y],
                    "freshest_t": row.freshest_t,
                    "detections": row.detections,
                }
                for row in sensor_fusions
            ],
            "n": len(dets),
        }
        return TrackEnvelope(
            track_id=track_id,
            t_min=min(ts),
            t_max=max(ts),
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            mean_confidence=mean_confidence,
            residual_uncertainty=residual,
            fingerprint=digest(body),
            fused_x=fused_x,
            fused_y=fused_y,
            effective_confidence=effective_confidence,
            residual_unknown_mass=residual_unknown_mass,
            sensor_count=len(sensor_fusions),
            reference_time=reference_time,
            sensor_fusions=tuple(sensor_fusions),
        )
