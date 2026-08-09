from __future__ import annotations
import unittest
from src.envelope import Detection, TrackEnvelopeCompiler

class Adv(unittest.TestCase):
    def test_bad_confidence(self):
        with self.assertRaises(ValueError):
            Detection("s1", 0.0, 0.0, 0.0, 1.5)
    def test_empty_none(self):
        self.assertIsNone(TrackEnvelopeCompiler().compile("T", []))
    def test_fingerprint_stable(self):
        dets = [Detection("s1", 0.0, 1.0, 2.0, 0.9)]
        a = TrackEnvelopeCompiler(pad=0.5).compile("T1", dets)
        b = TrackEnvelopeCompiler(pad=0.5).compile("T1", dets)
        assert a and b
        self.assertEqual(a.fingerprint, b.fingerprint)

if __name__ == "__main__":
    unittest.main()
