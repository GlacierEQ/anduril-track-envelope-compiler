#ifndef ENVELOPE_H
#define ENVELOPE_H

typedef struct {
    float t, x, y, confidence;
    int sensor_id; /* 0 preserves legacy behavior: each detection is independent. */
    float covariance_xx, covariance_yy;
} Detection;

typedef struct {
    float t_min, t_max, x_min, x_max, y_min, y_max;
    float mean_confidence, residual_uncertainty;
    int n;
    float fused_x, fused_y;
    float effective_confidence, residual_unknown_mass;
    int sensor_count;
    float reference_time;
} TrackEnvelope;

int envelope_compile(const Detection *dets, int n, float pad, TrackEnvelope *out);
int envelope_compile_v2(
    const Detection *dets,
    int n,
    float pad,
    float half_life,
    float sigma_scale,
    float age_covariance_growth,
    float reference_time,
    TrackEnvelope *out
);

#endif
