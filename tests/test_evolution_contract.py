import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = json.loads((ROOT / "machine" / "excellence-state.json").read_text(encoding="utf-8"))
POSITION = json.loads((ROOT / "machine" / "canonical-position.json").read_text(encoding="utf-8"))
TARGET = json.loads((ROOT / "machine" / "target-contract.json").read_text(encoding="utf-8"))
RECEIPT_PATH = ROOT / "machine" / "evolution-receipts" / "2026-08-11-temporal-covariance-fusion.json"
RECEIPT = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


class EvolutionContractTests(unittest.TestCase):
    def test_consumed_cursor_is_receipt_bound(self):
        self.assertEqual(RECEIPT["result"], "PASS")
        self.assertEqual(RECEIPT["candidate_source_sha"], "fb8e460e3b76b9d0453e702dfd2bd167368dd6a5")
        self.assertEqual(RECEIPT["workflow_run"], 31461547751)
        self.assertEqual(STATE["evolution_history"][-1]["receipt"], str(RECEIPT_PATH.relative_to(ROOT)))
        self.assertEqual(STATE["evolution_history"][-1]["consumed_cursor"], RECEIPT["consumed_cursor"])

    def test_next_cursor_is_consistent_across_machine_truth(self):
        expected = "next:authenticated_sensor_identity_correlation_domains_and_versioned_fusion_receipts"
        self.assertEqual(STATE["evolution_cursor"], expected)
        self.assertEqual(RECEIPT["next_cursor"], expected)
        self.assertEqual(TARGET["next_evolution"], expected)
        self.assertIn("Authenticate sensor identity", POSITION["next_evolution"])

    def test_external_claim_ceiling_did_not_inflate(self):
        self.assertEqual(STATE["claim_ceiling"], "PROMOTED")
        joined = " ".join(TARGET["nonclaims"]).lower()
        self.assertIn("no anduril affiliation", joined)
        self.assertIn("no production tracking deployment", joined)
        self.assertIn("no sensor identity authentication", joined)


if __name__ == "__main__":
    unittest.main()
