/* Babel: C — zero-alloc time-decayed covariance envelope hot path. */
#include "envelope.h"
#include <math.h>

static float fminf2(float a, float b) { return a < b ? a : b; }
static float fmaxf2(float a, float b) { return a > b ? a : b; }

static float decay(float age, float half_life) {
    return expf(-0.6931471805599453f * age / half_life);
}

int envelope_compile_v2(
    const Detection *dets,
    int n,
    float pad,
    float half_life,
    float sigma_scale,
    float age_covariance_growth,
    float reference_time,
    TrackEnvelope *out
) {
    if (!dets || n <= 0 || !out || pad < 0.f || half_life <= 0.f ||
        sigma_scale < 0.f || age_covariance_growth < 0.f || !isfinite(reference_time)) return -1;

    float xmin = INFINITY, xmax = -INFINITY;
    float ymin = INFINITY, ymax = -INFINITY;
    float tmin = dets[0].t, tmax = dets[0].t;
    float csum = 0.f, weighted_x = 0.f, weighted_y = 0.f, weight_sum = 0.f;
    float residual_unknown = 1.f, sigma_sum = 0.f;
    int sensor_count = 0;

    for (int i = 0; i < n; i++) {
        const Detection *d = &dets[i];
        if (!isfinite(d->t) || !isfinite(d->x) || !isfinite(d->y) ||
            !isfinite(d->confidence) || !isfinite(d->covariance_xx) || !isfinite(d->covariance_yy) ||
            d->confidence < 0.f || d->confidence > 1.f || d->covariance_xx < 0.f || d->covariance_yy < 0.f ||
            d->t > reference_time) return -2;

        float age = reference_time - d->t;
        float temporal = decay(age, half_life);
        float support = d->confidence * temporal;
        float age_growth = 1.f + age_covariance_growth * age / half_life;
        float sx = sqrtf(d->covariance_xx) * age_growth;
        float sy = sqrtf(d->covariance_yy) * age_growth;
        xmin = fminf2(xmin, d->x - pad - sigma_scale * sx);
        xmax = fmaxf2(xmax, d->x + pad + sigma_scale * sx);
        ymin = fminf2(ymin, d->y - pad - sigma_scale * sy);
        ymax = fmaxf2(ymax, d->y + pad + sigma_scale * sy);
        tmin = fminf2(tmin, d->t);
        tmax = fmaxf2(tmax, d->t);
        csum += d->confidence;
        weighted_x += d->x * support;
        weighted_y += d->y * support;
        weight_sum += support;
        sigma_sum += sx + sy;

        int first_for_sensor = 1;
        if (d->sensor_id != 0) {
            for (int j = 0; j < i; j++) {
                if (dets[j].sensor_id == d->sensor_id) {
                    first_for_sensor = 0;
                    break;
                }
            }
        }
        if (first_for_sensor) {
            float max_support = support;
            if (d->sensor_id != 0) {
                for (int j = i + 1; j < n; j++) {
                    if (dets[j].sensor_id == d->sensor_id) {
                        float other = dets[j].confidence * decay(reference_time - dets[j].t, half_life);
                        if (other > max_support) max_support = other;
                    }
                }
            }
            residual_unknown *= 1.f - max_support;
            sensor_count++;
        }
    }

    float mean_c = csum / (float)n;
    float fused_x = weight_sum > 0.f ? weighted_x / weight_sum : 0.f;
    float fused_y = weight_sum > 0.f ? weighted_y / weight_sum : 0.f;
    float effective_c = 1.f - residual_unknown;
    float spread = (xmax - xmin) + (ymax - ymin);
    float avg_sigma = sigma_sum / (2.f * (float)n);
    float residual = residual_unknown + tanhf(spread / 100.f) * 0.1f + tanhf(avg_sigma / 10.f) * 0.1f;
    if (residual > 1.f) residual = 1.f;

    out->t_min = tmin; out->t_max = tmax;
    out->x_min = xmin; out->x_max = xmax;
    out->y_min = ymin; out->y_max = ymax;
    out->mean_confidence = mean_c;
    out->residual_uncertainty = residual;
    out->n = n;
    out->fused_x = fused_x; out->fused_y = fused_y;
    out->effective_confidence = effective_c;
    out->residual_unknown_mass = residual_unknown;
    out->sensor_count = sensor_count;
    out->reference_time = reference_time;
    return 0;
}

int envelope_compile(const Detection *dets, int n, float pad, TrackEnvelope *out) {
    if (!dets || n <= 0) return -1;
    float reference_time = dets[0].t;
    for (int i = 1; i < n; i++) reference_time = fmaxf2(reference_time, dets[i].t);
    return envelope_compile_v2(dets, n, pad, 10.f, 2.f, 0.25f, reference_time, out);
}
