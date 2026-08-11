#include "envelope.h"
#include "envelope.c"
#include <stdio.h>
#include <math.h>

static int closef(float a, float b, float tol) { return fabsf(a - b) <= tol; }

int main(void) {
    TrackEnvelope e;

    /* Legacy four-field aggregate initialization remains valid and zero-fills new fields. */
    Detection legacy[2] = {{0,0,0,0.9f},{1,2,4,0.8f}};
    if (envelope_compile(legacy, 2, 1.f, &e) != 0) return 1;
    if (!closef(e.x_min, -1.f, 1e-5f)) { printf("legacy xmin %f\n", e.x_min); return 2; }
    if (!closef(e.x_max, 3.f, 1e-5f)) { printf("legacy xmax %f\n", e.x_max); return 3; }
    if (e.residual_uncertainty <= 0.f || e.sensor_count != 2) return 4;
    if (envelope_compile(legacy, 0, 1.f, &e) != -1) return 5;

    Detection fresh[1] = {{20,5,5,0.8f,1,0,0}};
    TrackEnvelope fresh_e;
    if (envelope_compile_v2(fresh, 1, 0.f, 10.f, 2.f, 0.25f, 20.f, &fresh_e) != 0) return 6;
    Detection stale[1] = {{10,5,5,0.8f,1,0,0}};
    TrackEnvelope stale_e;
    if (envelope_compile_v2(stale, 1, 0.f, 10.f, 2.f, 0.25f, 20.f, &stale_e) != 0) return 7;
    if (!(fresh_e.effective_confidence > stale_e.effective_confidence)) return 8;
    if (!(fresh_e.residual_unknown_mass < stale_e.residual_unknown_mass)) return 9;
    if (!closef(fresh_e.residual_unknown_mass, 0.2f, 1e-5f)) return 10;
    if (!closef(stale_e.residual_unknown_mass, 0.6f, 1e-5f)) return 11;

    Detection point[1] = {{20,10,-2,0.9f,1,0,0}};
    Detection cov[1] = {{20,10,-2,0.9f,1,4,9}};
    TrackEnvelope point_e, cov_e;
    if (envelope_compile_v2(point, 1, 1.f, 10.f, 2.f, 0.5f, 20.f, &point_e) != 0) return 12;
    if (envelope_compile_v2(cov, 1, 1.f, 10.f, 2.f, 0.5f, 20.f, &cov_e) != 0) return 13;
    if (!(cov_e.x_min < point_e.x_min && cov_e.x_max > point_e.x_max)) return 14;
    if (!(cov_e.y_min < point_e.y_min && cov_e.y_max > point_e.y_max)) return 15;

    Detection same[2] = {
        {10,0,0,0.8f,7,0,0},
        {10,2,0,0.8f,7,0,0}
    };
    Detection independent[2] = {
        {10,0,0,0.8f,7,0,0},
        {10,2,0,0.8f,8,0,0}
    };
    TrackEnvelope same_e, independent_e;
    if (envelope_compile_v2(same, 2, 0.f, 10.f, 2.f, 0.25f, 10.f, &same_e) != 0) return 16;
    if (envelope_compile_v2(independent, 2, 0.f, 10.f, 2.f, 0.25f, 10.f, &independent_e) != 0) return 17;
    if (same_e.sensor_count != 1 || independent_e.sensor_count != 2) return 18;
    if (!(same_e.residual_unknown_mass > independent_e.residual_unknown_mass)) return 19;
    if (!closef(same_e.residual_unknown_mass, 0.2f, 1e-5f)) return 20;
    if (!closef(independent_e.residual_unknown_mass, 0.04f, 1e-5f)) return 21;

    Detection fused[2] = {
        {0,0,0,1.f,1,0,0},
        {10,10,0,1.f,2,0,0}
    };
    TrackEnvelope fused_e;
    if (envelope_compile_v2(fused, 2, 0.f, 10.f, 2.f, 0.25f, 10.f, &fused_e) != 0) return 22;
    if (!(fused_e.fused_x > 5.f && fused_e.fused_x < 10.f)) return 23;

    Detection future[1] = {{21,0,0,0.8f,1,0,0}};
    if (envelope_compile_v2(future, 1, 0.f, 10.f, 2.f, 0.25f, 20.f, &e) != -2) return 24;

    printf("ok\n");
    return 0;
}
