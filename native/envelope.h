#ifndef ENVELOPE_H
#define ENVELOPE_H

#include <stdint.h>

typedef struct { float t, x, y, confidence; } Detection;

typedef struct {
    uint32_t sensor_id;
    float t, x, y, confidence;
    float var_x, var_y, cov_xy;
} DetectionV2;

typedef struct {
    float t_min, t_max, x_min, x_max, y_min, y_max;
    float mean_confidence, fused_confidence;
    float residual_unknown_mass, residual_uncertainty;
    int sensor_count;
    int n;
} TrackEnvelope;

int envelope_compile(const Detection *dets, int n, float pad, TrackEnvelope *out);
int envelope_compile_v2(
    const DetectionV2 *dets,
    int n,
    float pad,
    float half_life,
    float covariance_sigma,
    TrackEnvelope *out
);

#endif
