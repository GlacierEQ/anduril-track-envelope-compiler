from __future__ import annotations

import unittest

from src.envelope import Detection, TrackEnvelopeCompiler


class EnvTests(unittest.TestCase):
    def test_empty_none(self):
        self.assertIsNone(TrackEnvelopeCompiler().compile("T1", []))

    def test_legacy_envelope_bounds_remain_compatible(self):
        dets = [
            Detection("s1", 0.0, 0.0, 0.0, 0.9),
            Detection("s2", 1.0, 2.0, 4.0, 0.8),
        ]
        env = TrackEnvelopeCompiler(pad=1.0).compile("T1", dets)
        assert env is not None
        self.assertEqual(env.x_min, -1.0)
        self.assertEqual(env.x_max, 3.0)
        self.assertGreater(env.residual_uncertainty, 0.0)
        self.assertEqual(env.sensor_count, 2)
        self.assertEqual(len(env.fingerprint), 64)

    def test_stale_detection_loses_support_at_fixed_reference_time(self):
        compiler = TrackEnvelopeCompiler(pad=0.0, half_life=10.0)
        fresh = compiler.compile(
            "T",
            [Detection("s1", 20.0, 5.0, 5.0, 0.8)],
            reference_time=20.0,
        )
        stale = compiler.compile(
            "T",
            [Detection("s1", 10.0, 5.0, 5.0, 0.8)],
            reference_time=20.0,
        )
        assert fresh is not None and stale is not None
        self.assertGreater(fresh.effective_confidence, stale.effective_confidence)
        self.assertLess(fresh.residual_unknown_mass, stale.residual_unknown_mass)
        self.assertAlmostEqual(fresh.residual_unknown_mass, 0.2)
        self.assertAlmostEqual(stale.residual_unknown_mass, 0.6)

    def test_covariance_expands_bounds_and_age_inflates_it_further(self):
        compiler = TrackEnvelopeCompiler(
            pad=1.0,
            half_life=10.0,
            sigma_scale=2.0,
            age_covariance_growth=0.5,
        )
        point = Detection("s1", 20.0, 10.0, -2.0, 0.9)
        uncertain = Detection(
            "s1",
            20.0,
            10.0,
            -2.0,
            0.9,
            covariance_xx=4.0,
            covariance_yy=9.0,
        )
        stale_uncertain = Detection(
            "s1",
            10.0,
            10.0,
            -2.0,
            0.9,
            covariance_xx=4.0,
            covariance_yy=9.0,
        )
        base = compiler.compile("T", [point], reference_time=20.0)
        current = compiler.compile("T", [uncertain], reference_time=20.0)
        stale = compiler.compile("T", [stale_uncertain], reference_time=20.0)
        assert base is not None and current is not None and stale is not None
        self.assertLess(current.x_min, base.x_min)
        self.assertGreater(current.x_max, base.x_max)
        self.assertLess(current.y_min, base.y_min)
        self.assertGreater(current.y_max, base.y_max)
        self.assertLess(stale.x_min, current.x_min)
        self.assertGreater(stale.y_max, current.y_max)

    def test_repeated_same_sensor_does_not_manufacture_independent_known_mass(self):
        compiler = TrackEnvelopeCompiler(pad=0.0)
        same_sensor = compiler.compile(
            "T",
            [
                Detection("s1", 10.0, 0.0, 0.0, 0.8),
                Detection("s1", 10.0, 2.0, 0.0, 0.8),
            ],
            reference_time=10.0,
        )
        independent = compiler.compile(
            "T",
            [
                Detection("s1", 10.0, 0.0, 0.0, 0.8),
                Detection("s2", 10.0, 2.0, 0.0, 0.8),
            ],
            reference_time=10.0,
        )
        assert same_sensor is not None and independent is not None
        self.assertEqual(same_sensor.sensor_count, 1)
        self.assertEqual(independent.sensor_count, 2)
        self.assertAlmostEqual(same_sensor.residual_unknown_mass, 0.2)
        self.assertAlmostEqual(independent.residual_unknown_mass, 0.04)
        self.assertGreater(
            same_sensor.residual_unknown_mass,
            independent.residual_unknown_mass,
        )

    def test_unknown_mass_is_preserved_unless_sensor_support_is_exactly_one(self):
        compiler = TrackEnvelopeCompiler(pad=0.0)
        bounded = compiler.compile(
            "T",
            [
                Detection("s1", 0.0, 0.0, 0.0, 0.99),
                Detection("s2", 0.0, 0.0, 0.0, 0.99),
            ],
            reference_time=0.0,
        )
        certain = compiler.compile(
            "T",
            [Detection("s1", 0.0, 0.0, 0.0, 1.0)],
            reference_time=0.0,
        )
        assert bounded is not None and certain is not None
        self.assertGreater(bounded.residual_unknown_mass, 0.0)
        self.assertLess(bounded.effective_confidence, 1.0)
        self.assertEqual(certain.residual_unknown_mass, 0.0)
        self.assertEqual(certain.effective_confidence, 1.0)

    def test_fused_center_moves_toward_fresher_supported_sensor(self):
        compiler = TrackEnvelopeCompiler(pad=0.0, half_life=10.0)
        env = compiler.compile(
            "T",
            [
                Detection("old", 0.0, 0.0, 0.0, 1.0),
                Detection("fresh", 10.0, 10.0, 0.0, 1.0),
            ],
            reference_time=10.0,
        )
        assert env is not None
        self.assertGreater(env.fused_x, 5.0)
        self.assertLess(env.fused_x, 10.0)
        self.assertAlmostEqual(env.fused_y, 0.0)

    def test_invalid_covariance_and_reference_time_fail_closed(self):
        with self.assertRaises(ValueError):
            Detection(
                "s1",
                0.0,
                0.0,
                0.0,
                0.8,
                covariance_xx=1.0,
                covariance_yy=1.0,
                covariance_xy=2.0,
            )
        compiler = TrackEnvelopeCompiler()
        with self.assertRaises(ValueError):
            compiler.compile(
                "T",
                [Detection("s1", 10.0, 0.0, 0.0, 0.8)],
                reference_time=9.0,
            )


if __name__ == "__main__":
    unittest.main()
