#ifndef ENVELOPE_H
#define ENVELOPE_H
typedef struct { float t, x, y, confidence; } Detection;
typedef struct {
    float t_min, t_max, x_min, x_max, y_min, y_max;
    float mean_confidence, residual_uncertainty;
    int n;
} TrackEnvelope;
int envelope_compile(const Detection *dets, int n, float pad, TrackEnvelope *out);
#endif
