from __future__ import annotations
import unittest
from src.envelope import Detection, TrackEnvelopeCompiler

class EnvTests(unittest.TestCase):
    def test_empty_none(self):
        self.assertIsNone(TrackEnvelopeCompiler().compile("T1", []))

    def test_envelope_bounds(self):
        dets = [
            Detection("s1", 0.0, 0.0, 0.0, 0.9),
            Detection("s2", 1.0, 2.0, 4.0, 0.8),
        ]
        env = TrackEnvelopeCompiler(pad=1.0).compile("T1", dets)
        assert env is not None
        self.assertEqual(env.x_min, -1.0)
        self.assertEqual(env.x_max, 3.0)
        self.assertGreater(env.residual_uncertainty, 0.0)
        self.assertEqual(len(env.fingerprint), 64)

if __name__ == "__main__":
    unittest.main()
