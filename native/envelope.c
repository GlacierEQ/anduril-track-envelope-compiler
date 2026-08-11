/* Babel: C — zero-alloc time-decayed covariance envelope hot path. */
#include "envelope.h"
#include <math.h>

static float fminf2(float a, float b) { return a < b ? a : b; }
static float fmaxf2(float a, float b) { return a > b ? a : b; }

static float decay(float age, float half_life) {
    return expf(-0.6931471805599453f * age / half_life);
}

static int same_sensor(const Detection *a, const Detection *b, int ai, int bi) {
    /* sensor_id==0 is the backward-compatible legacy mode: each detection is independent. */
    if (a->sensor_id == 0 || b->sensor_id == 0) return ai == bi;
    return a->sensor_id == b->sensor_id;
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
    if (!dets || n <= 0 || !out || !isfinite(pad) || !isfinite(half_life) ||
        !isfinite(sigma_scale) || !isfinite(age_covariance_growth) ||
        pad < 0.f || half_life <= 0.f || sigma_scale < 0.f ||
        age_covariance_growth < 0.f || !isfinite(reference_time)) return -1;

    float xmin = INFINITY, xmax = -INFINITY;
    float ymin = INFINITY, ymax = -INFINITY;
    float tmin = dets[0].t, tmax = dets[0].t;
    float csum = 0.f, sigma_sum = 0.f;
    float residual_unknown = 1.f;
    int sensor_count = 0;

    /* Geometry and input validation remain detection-granular. */
    for (int i = 0; i < n; i++) {
        const Detection *d = &dets[i];
        if (!isfinite(d->t) || !isfinite(d->x) || !isfinite(d->y) ||
            !isfinite(d->confidence) || !isfinite(d->covariance_xx) || !isfinite(d->covariance_yy) ||
            d->confidence < 0.f || d->confidence > 1.f || d->covariance_xx < 0.f || d->covariance_yy < 0.f ||
            d->t > reference_time) return -2;

        float age = reference_time - d->t;
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
        sigma_sum += sx + sy;
    }

    /* Independent sensor fusion. Repeated reports refine one sensor centroid but
       cannot multiply that sensor's known-mass contribution. O(n^2), zero alloc. */
    float fusion_weight_sum = 0.f, fusion_x_sum = 0.f, fusion_y_sum = 0.f;
    for (int i = 0; i < n; i++) {
        int first_for_sensor = 1;
        for (int j = 0; j < i; j++) {
            if (same_sensor(&dets[i], &dets[j], i, j)) {
                first_for_sensor = 0;
                break;
            }
        }
        if (!first_for_sensor) continue;

        float member_weight_sum = 0.f;
        float member_x_sum = 0.f, member_y_sum = 0.f;
        float max_support = 0.f;
        for (int j = i; j < n; j++) {
            if (!same_sensor(&dets[i], &dets[j], i, j)) continue;
            float age = reference_time - dets[j].t;
            float support = dets[j].confidence * decay(age, half_life);
            member_weight_sum += support;
            member_x_sum += dets[j].x * support;
            member_y_sum += dets[j].y * support;
            if (support > max_support) max_support = support;
        }
        float sensor_x = dets[i].x;
        float sensor_y = dets[i].y;
        if (member_weight_sum > 0.f) {
            sensor_x = member_x_sum / member_weight_sum;
            sensor_y = member_y_sum / member_weight_sum;
        }
        residual_unknown *= 1.f - max_support;
        fusion_weight_sum += max_support;
        fusion_x_sum += sensor_x * max_support;
        fusion_y_sum += sensor_y * max_support;
        sensor_count++;
    }

    float mean_c = csum / (float)n;
    float fused_x, fused_y;
    if (fusion_weight_sum > 0.f) {
        fused_x = fusion_x_sum / fusion_weight_sum;
        fused_y = fusion_y_sum / fusion_weight_sum;
    } else {
        float centroid_x = 0.f, centroid_y = 0.f;
        for (int i = 0; i < n; i++) {
            centroid_x += dets[i].x;
            centroid_y += dets[i].y;
        }
        fused_x = centroid_x / (float)n;
        fused_y = centroid_y / (float)n;
    }
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
