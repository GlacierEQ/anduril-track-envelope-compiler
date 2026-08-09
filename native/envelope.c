/* Babel: C — zero-alloc detection envelope bounds (hot geometry path). */
#include "envelope.h"
#include <math.h>

static float fminf2(float a, float b) { return a < b ? a : b; }
static float fmaxf2(float a, float b) { return a > b ? a : b; }

int envelope_compile(const Detection *dets, int n, float pad, TrackEnvelope *out) {
    if (!dets || n <= 0 || !out) return -1;
    float xmin = dets[0].x, xmax = dets[0].x;
    float ymin = dets[0].y, ymax = dets[0].y;
    float tmin = dets[0].t, tmax = dets[0].t;
    float csum = 0.f;
    for (int i = 0; i < n; i++) {
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
    out->residual_uncertainty = residual;
    out->n = n;
    return 0;
}
