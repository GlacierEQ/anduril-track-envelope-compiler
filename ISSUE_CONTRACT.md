# ISSUE CONTRACT

## Pain
Point-like tracks and normalized confidence can hide stale evidence, repeated observations from one sensor, covariance growth, and the probability mass that remains unknown.

## Success
- Compile detections into a spatiotemporal envelope rather than a point-only track.
- Decay evidence support with age under an explicit half-life policy.
- Prevent repeated observations from one sensor from counting as independent confidence.
- Expand spatial bounds from valid covariance instead of discarding it.
- Preserve residual unknown mass as first-class output.
- Reject empty evidence, malformed/non-finite evidence, and non-PSD covariance.
- Bind deterministic Python envelope identity to exact evidence plus fusion policy.

## Boundaries
- Sensor IDs are caller-supplied labels, not authenticated identities.
- Distinct sensor IDs are treated as independent evidence by this reference fusion rule; cross-sensor correlation provenance is not yet implemented.
- No Anduril affiliation, adoption, production tracking deployment, or operational targeting authority is claimed.
