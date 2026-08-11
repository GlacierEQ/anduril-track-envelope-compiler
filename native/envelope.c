/* Babel: C — zero-allocation envelope geometry and uncertainty fusion. */
#include "envelope.h"
#include <math.h>

static float fminf2(float a, float b) { return a < b ? a : b; }
static float fmaxf2(float a, float b) { return a > b ? a : b; }

static int finite_nonnegative(float value) {
    return isfinite(value) && value >= 0.f;
}

int envelope_compile(const Detection *dets, int n, float pad, TrackEnvelope *out) {
    if (!dets || n <= 0 || !out || !finite_nonnegative(pad)) return -1;
    float xmin = dets[0].x, xmax = dets[0].x;
    float ymin = dets[0].y, ymax = dets[0].y;
    float tmin = dets[0].t, tmax = dets[0].t;
    float csum = 0.f;
    for (int i = 0; i < n; i++) {
        if (!isfinite(dets[i].t) || !isfinite(dets[i].x) || !isfinite(dets[i].y) ||
            !isfinite(dets[i].confidence) || dets[i].confidence < 0.f || dets[i].confidence > 1.f) {
            return -2;
        }
        xmin = fminf2(xmin, dets[i].x);
        xmax = fmaxf2(xmax, dets[i].x);
        ymin = fminf2(ymin, dets[i].y);
        ymax = fmaxf2(ymax, dets[i].y);
        tmin = fminf2(tmin, dets[i].t);
        tmax = fmaxf2(tmax, dets[i].t);
        csum += dets[i].confidence;
    }
    float mean_c = csum / (float)n;
    float spread = (xmax - xmin) + (ymax - ymin);
    float residual = (1.f - mean_c) + tanhf(spread / 100.f) * 0.2f;
    if (residual > 1.f) residual = 1.f;
    out->t_min = tmin; out->t_max = tmax;
    out->x_min = xmin - pad; out->x_max = xmax + pad;
    out->y_min = ymin - pad; out->y_max = ymax + pad;
    out->mean_confidence = mean_c;
    out->fused_confidence = mean_c;
    out->residual_unknown_mass = 1.f - mean_c;
    out->residual_uncertainty = residual;
    out->sensor_count = n;
    out->n = n;
    return 0;
}

int envelope_compile_v2(
    const DetectionV2 *dets,
    int n,
    float pad,
    float half_life,
    float covariance_sigma,
    TrackEnvelope *out
) {
    if (!dets || n <= 0 || !out || !finite_nonnegative(pad) ||
        !isfinite(half_life) || half_life <= 0.f || !finite_nonnegative(covariance_sigma)) {
        return -1;
    }

    float raw_xmin = dets[0].x, raw_xmax = dets[0].x;
    float raw_ymin = dets[0].y, raw_ymax = dets[0].y;
    float tmin = dets[0].t, tmax = dets[0].t;
    float xmin = 0.f, xmax = 0.f, ymin = 0.f, ymax = 0.f;
    float csum = 0.f, sigma_sum = 0.f;

    for (int i = 0; i < n; i++) {
        const DetectionV2 *d = &dets[i];
        if (d->sensor_id == 0 || !isfinite(d->t) || !isfinite(d->x) || !isfinite(d->y) ||
            !isfinite(d->confidence) || d->confidence < 0.f || d->confidence > 1.f ||
            !finite_nonnegative(d->var_x) || !finite_nonnegative(d->var_y) || !isfinite(d->cov_xy)) {
            return -2;
        }
        float determinant = d->var_x * d->var_y - d->cov_xy * d->cov_xy;
        if (determinant < -1e-5f) return -3;

        float sigma_x = sqrtf(d->var_x);
        float sigma_y = sqrtf(d->var_y);
        float xgrowth = pad + covariance_sigma * sigma_x;
        float ygrowth = pad + covariance_sigma * sigma_y;
        float dlo_x = d->x - xgrowth, dhi_x = d->x + xgrowth;
        float dlo_y = d->y - ygrowth, dhi_y = d->y + ygrowth;
        if (i == 0) {
            xmin = dlo_x; xmax = dhi_x; ymin = dlo_y; ymax = dhi_y;
        } else {
            xmin = fminf2(xmin, dlo_x); xmax = fmaxf2(xmax, dhi_x);
            ymin = fminf2(ymin, dlo_y); ymax = fmaxf2(ymax, dhi_y);
        }
        raw_xmin = fminf2(raw_xmin, d->x); raw_xmax = fmaxf2(raw_xmax, d->x);
        raw_ymin = fminf2(raw_ymin, d->y); raw_ymax = fmaxf2(raw_ymax, d->y);
        tmin = fminf2(tmin, d->t); tmax = fmaxf2(tmax, d->t);
        csum += d->confidence;
        sigma_sum += sqrtf(d->var_x + d->var_y);
    }

    float residual_unknown = 1.f;
    int sensor_count = 0;
    for (int i = 0; i < n; i++) {
        int first = 1;
        for (int j = 0; j < i; j++) {
            if (dets[j].sensor_id == dets[i].sensor_id) {
                first = 0;
                break;
            }
        }
        if (!first) continue;
        sensor_count++;
        float best_support = 0.f;
        for (int j = i; j < n; j++) {
            if (dets[j].sensor_id != dets[i].sensor_id) continue;
            float age = tmax - dets[j].t;
            float decay = powf(0.5f, age / half_life);
            float support = dets[j].confidence * decay;
            if (support > best_support) best_support = support;
        }
        residual_unknown *= 1.f - best_support;
    }
    if (residual_unknown < 0.f) residual_unknown = 0.f;
    if (residual_unknown > 1.f) residual_unknown = 1.f;

    float spread = (raw_xmax - raw_xmin) + (raw_ymax - raw_ymin);
    float spatial_penalty = tanhf(spread / 100.f) * 0.15f;
    float mean_sigma = sigma_sum / (float)n;
    float covariance_penalty = tanhf(mean_sigma / 10.f) * 0.15f;
    float residual = residual_unknown + spatial_penalty + covariance_penalty;
    if (residual > 1.f) residual = 1.f;

    out->t_min = tmin; out->t_max = tmax;
    out->x_min = xmin; out->x_max = xmax;
    out->y_min = ymin; out->y_max = ymax;
    out->mean_confidence = csum / (float)n;
    out->fused_confidence = 1.f - residual_unknown;
    out->residual_unknown_mass = residual_unknown;
    out->residual_uncertainty = residual;
    out->sensor_count = sensor_count;
    out->n = n;
    return 0;
}
