#!/usr/bin/env python3
"""Cold-start: TrackEnvelopeCompiler bounds on real detections."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from envelope import Detection, TrackEnvelopeCompiler

def main() -> int:
    dets = [
        Detection("s1", 0.0, 0.0, 0.0, 0.9),
        Detection("s2", 1.0, 2.0, 4.0, 0.8),
    ]
    env = TrackEnvelopeCompiler(pad=1.0).compile("T1", dets)
    ok = (
        env is not None
        and env.x_min == -1.0
        and env.x_max == 3.0
        and env.residual_uncertainty > 0.0
        and len(env.fingerprint) == 64
    )
    out = {
        "track_id": None if env is None else env.track_id,
        "x_min": None if env is None else env.x_min,
        "x_max": None if env is None else env.x_max,
        "residual_uncertainty": None if env is None else env.residual_uncertainty,
        "fingerprint": None if env is None else env.fingerprint,
        "expected_x_min": -1.0,
        "expected_x_max": 3.0,
        "ok": ok,
    }
    print(json.dumps(out, sort_keys=True))
    return 0 if ok else 1
if __name__ == "__main__":
    raise SystemExit(main())
