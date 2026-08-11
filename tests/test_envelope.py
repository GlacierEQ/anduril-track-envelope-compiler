from __future__ import annotations
import math
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
        self.assertEqual(len(env.policy_fingerprint), 64)

    def test_repeated_sensor_does_not_fake_independent_confidence(self):
        compiler = TrackEnvelopeCompiler()
        same = compiler.compile(
            "T1",
            [
                Detection("s1", 10.0, 0.0, 0.0, 0.9),
                Detection("s1", 10.0, 1.0, 0.0, 0.9),
            ],
        )
        distinct = compiler.compile(
            "T1",
            [
                Detection("s1", 10.0, 0.0, 0.0, 0.9),
                Detection("s2", 10.0, 1.0, 0.0, 0.9),
            ],
        )
        assert same is not None and distinct is not None
        self.assertEqual(same.sensor_count, 1)
        self.assertAlmostEqual(same.fused_confidence, 0.9)
        self.assertAlmostEqual(same.residual_unknown_mass, 0.1)
        self.assertEqual(distinct.sensor_count, 2)
        self.assertAlmostEqual(distinct.residual_unknown_mass, 0.01)
        self.assertGreater(distinct.fused_confidence, same.fused_confidence)

    def test_temporal_decay_preserves_stale_unknown_mass(self):
        fresh = TrackEnvelopeCompiler(half_life=1.0).compile(
            "T1",
            [
                Detection("s1", 10.0, 0.0, 0.0, 0.9),
                Detection("s2", 10.0, 1.0, 0.0, 0.9),
            ],
        )
        stale = TrackEnvelopeCompiler(half_life=1.0).compile(
            "T1",
            [
                Detection("s1", 0.0, 0.0, 0.0, 0.9),
                Detection("s2", 10.0, 1.0, 0.0, 0.9),
            ],
        )
        assert fresh is not None and stale is not None
        self.assertLess(stale.fused_confidence, fresh.fused_confidence)
        self.assertGreater(stale.residual_unknown_mass, fresh.residual_unknown_mass)

    def test_covariance_expands_bounds_and_uncertainty(self):
        env = TrackEnvelopeCompiler(pad=1.0, covariance_sigma=2.0).compile(
            "T1",
            [Detection("s1", 5.0, 0.0, 0.0, 0.9, var_x=4.0, var_y=9.0)],
        )
        assert env is not None
        self.assertEqual(env.x_min, -5.0)
        self.assertEqual(env.x_max, 5.0)
        self.assertEqual(env.y_min, -7.0)
        self.assertEqual(env.y_max, 7.0)
        self.assertGreater(env.residual_uncertainty, env.residual_unknown_mass)

    def test_order_does_not_change_identity(self):
        dets = [
            Detection("s2", 2.0, 3.0, 4.0, 0.7, 1.0, 1.0, 0.25),
            Detection("s1", 1.0, 1.0, 2.0, 0.8, 0.5, 0.5, 0.0),
        ]
        compiler = TrackEnvelopeCompiler()
        a = compiler.compile("T1", dets)
        b = compiler.compile("T1", list(reversed(dets)))
        assert a is not None and b is not None
        self.assertEqual(a.fingerprint, b.fingerprint)

    def test_bad_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            Detection("", 0.0, 0.0, 0.0, 0.9)
        with self.assertRaises(ValueError):
            Detection("s1", 0.0, math.nan, 0.0, 0.9)
        with self.assertRaises(ValueError):
            Detection("s1", 0.0, 0.0, 0.0, 0.9, var_x=1.0, var_y=1.0, cov_xy=2.0)
        with self.assertRaises(ValueError):
            TrackEnvelopeCompiler(pad=-1.0)
        with self.assertRaises(ValueError):
            TrackEnvelopeCompiler(half_life=0.0)
        with self.assertRaises(ValueError):
            TrackEnvelopeCompiler(covariance_sigma=-1.0)
        with self.assertRaises(ValueError):
            TrackEnvelopeCompiler().compile("", [Detection("s1", 0.0, 0.0, 0.0, 0.9)])


if __name__ == "__main__":
    unittest.main()
