from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = json.loads((ROOT / "machine" / "excellence-state.json").read_text())
POSITION = json.loads((ROOT / "machine" / "canonical-position.json").read_text())
CONTRACT = json.loads((ROOT / "machine" / "target-contract.json").read_text())
RECEIPT = json.loads(
    (
        ROOT
        / "machine"
        / "evolution-receipts"
        / "2026-08-11-temporal-covariance-fusion.json"
    ).read_text()
)


class EvolutionReceiptContractTests(unittest.TestCase):
    def test_current_cursor_is_consumed_by_exact_green_candidate(self):
        self.assertEqual(
            RECEIPT["consumed_cursor"],
            "next:temporal_decay_covariance_aware_growth_multi_sensor_fusion_preserve_residual_unknown_mass",
        )
        self.assertEqual(
            RECEIPT["candidate_source_sha"],
            "fae1453719e54782c8c265635c8f05f121086995",
        )
        self.assertEqual(RECEIPT["workflow_run"], 31454664574)
        self.assertEqual(RECEIPT["proof"], {"python": "PASS", "native_c": "PASS"})

    def test_next_cursor_is_consistent_across_machine_surfaces(self):
        expected = (
            "next:cross_covariance_rotation_motion_prediction_outlier_gating_"
            "and_durable_fusion_receipt_chain"
        )
        self.assertEqual(STATE["evolution_cursor"], expected)
        self.assertEqual(POSITION["next_evolution_cursor"], expected)
        self.assertEqual(CONTRACT["target"]["next_cursor"], expected)
        self.assertEqual(RECEIPT["next_cursor"], expected)

    def test_cross_covariance_and_future_mechanisms_remain_nonclaims(self):
        text = " ".join(
            POSITION["nonclaims"]
            + CONTRACT["nonclaims"]
            + RECEIPT["truth_boundaries"]
        ).lower()
        self.assertIn("covariance_xy", text)
        self.assertIn("diagonal covariance", text)
        self.assertIn("motion", text)
        self.assertIn("outlier", text)
        self.assertIn("durable", text)

    def test_identity_and_unknown_mass_authority_are_preserved(self):
        self.assertEqual(POSITION["repository"], STATE["repository"])
        self.assertEqual(STATE["principal_state"], "EVOLVING")
        self.assertIn(
            "independent-sensor residual unknown-mass fusion",
            POSITION["capability_contribution"]["extractable_mechanisms"],
        )
        policy = POSITION["integration_policy"]
        self.assertTrue(policy["preserve_repository_identity"])
        self.assertTrue(policy["preserve_lineage"])


if __name__ == "__main__":
    unittest.main()
