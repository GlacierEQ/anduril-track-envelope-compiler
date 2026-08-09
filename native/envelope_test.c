#include "envelope.h"
#include "envelope.c"
#include <stdio.h>
#include <math.h>
int main(void) {
    Detection d[2] = {{0,0,0,0.9f},{1,2,4,0.8f}};
    TrackEnvelope e;
    if (envelope_compile(d, 2, 1.f, &e) != 0) return 1;
    if (fabsf(e.x_min + 1.f) > 1e-5f) { printf("xmin %f\n", e.x_min); return 2; }
    if (fabsf(e.x_max - 3.f) > 1e-5f) { printf("xmax %f\n", e.x_max); return 3; }
    if (e.residual_uncertainty <= 0.f) return 4;
    if (envelope_compile(d, 0, 1.f, &e) != -1) return 5;
    printf("ok\n");
    return 0;
}
