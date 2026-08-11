#include "envelope.h"
#include "envelope.c"
#include <math.h>
#include <stdio.h>

static int closef(float a, float b) { return fabsf(a - b) <= 1e-4f; }

int main(void) {
    Detection legacy[2] = {{0,0,0,0.9f},{1,2,4,0.8f}};
    TrackEnvelope e;
    if (envelope_compile(legacy, 2, 1.f, &e) != 0) return 1;
    if (!closef(e.x_min, -1.f) || !closef(e.x_max, 3.f)) return 2;
    if (e.residual_uncertainty <= 0.f) return 3;
    if (envelope_compile(legacy, 0, 1.f, &e) != -1) return 4;

    DetectionV2 same_sensor[2] = {
        {1, 10.f, 0.f, 0.f, 0.9f, 0.f, 0.f, 0.f},
        {1, 10.f, 1.f, 0.f, 0.9f, 0.f, 0.f, 0.f},
    };
    TrackEnvelope same;
    if (envelope_compile_v2(same_sensor, 2, 1.f, 5.f, 2.f, &same) != 0) return 5;
    if (same.sensor_count != 1 || !closef(same.fused_confidence, 0.9f)) return 6;
    if (!closef(same.residual_unknown_mass, 0.1f)) return 7;

    DetectionV2 distinct[2] = {
        {1, 10.f, 0.f, 0.f, 0.9f, 0.f, 0.f, 0.f},
        {2, 10.f, 1.f, 0.f, 0.9f, 0.f, 0.f, 0.f},
    };
    TrackEnvelope fused;
    if (envelope_compile_v2(distinct, 2, 1.f, 5.f, 2.f, &fused) != 0) return 8;
    if (fused.sensor_count != 2 || fused.fused_confidence <= same.fused_confidence) return 9;
    if (!closef(fused.residual_unknown_mass, 0.01f)) return 10;

    DetectionV2 stale[2] = {
        {1, 0.f, 0.f, 0.f, 0.9f, 0.f, 0.f, 0.f},
        {2, 10.f, 1.f, 0.f, 0.9f, 0.f, 0.f, 0.f},
    };
    TrackEnvelope decayed;
    if (envelope_compile_v2(stale, 2, 1.f, 1.f, 2.f, &decayed) != 0) return 11;
    if (!(decayed.fused_confidence < fused.fused_confidence)) return 12;

    DetectionV2 uncertain[1] = {
        {1, 5.f, 0.f, 0.f, 0.9f, 4.f, 9.f, 0.f},
    };
    TrackEnvelope cov;
    if (envelope_compile_v2(uncertain, 1, 1.f, 5.f, 2.f, &cov) != 0) return 13;
    if (!closef(cov.x_min, -5.f) || !closef(cov.x_max, 5.f)) return 14;
    if (!closef(cov.y_min, -7.f) || !closef(cov.y_max, 7.f)) return 15;
    if (!(cov.residual_uncertainty > cov.residual_unknown_mass)) return 16;

    DetectionV2 bad_cov[1] = {
        {1, 1.f, 0.f, 0.f, 0.9f, 1.f, 1.f, 2.f},
    };
    if (envelope_compile_v2(bad_cov, 1, 1.f, 5.f, 2.f, &e) != -3) return 17;
    DetectionV2 bad_nan[1] = {
        {1, 1.f, NAN, 0.f, 0.9f, 0.f, 0.f, 0.f},
    };
    if (envelope_compile_v2(bad_nan, 1, 1.f, 5.f, 2.f, &e) != -2) return 18;

    printf("ok\n");
    return 0;
}
