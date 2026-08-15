import unittest

import numpy as np

from ews import config
from ews.synth import generate
from ews.firewall import Vault
from ews.features import build_features, FEATURE_COLUMNS


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = generate(config.SEED)
        cls.vault = Vault(cls.panel.onset)
        cls.feats = build_features(cls.panel, cls.vault)

    def test_schema_and_no_nulls(self):
        expected = ["drug_idx", "anchor_month"] + FEATURE_COLUMNS + ["y"]
        self.assertEqual(list(self.feats.columns), expected)
        self.assertFalse(self.feats.isnull().any().any())

    def test_no_feature_equals_label(self):
        y = self.feats["y"].to_numpy()
        for col in FEATURE_COLUMNS:
            self.assertFalse(np.array_equal(self.feats[col].to_numpy(), y),
                             f"feature {col} equals the label")

    def test_anchor_rows_are_at_risk(self):
        # Every anchor month must be one where the drug is not already short.
        in_short = self.panel.frame.set_index(["drug_id", "month"])["in_shortage"]
        sample = self.feats.sample(min(200, len(self.feats)), random_state=0)
        for _, r in sample.iterrows():
            did = "D%04d" % int(r["drug_idx"])
            self.assertEqual(int(in_short.loc[(did, int(r["anchor_month"]))]), 0)

    def test_label_window_fully_observed(self):
        # No anchor should reach past the observed horizon.
        max_anchor = int(self.feats["anchor_month"].max())
        self.assertLess(max_anchor + config.LEAD_HORIZON, config.PERIOD_MONTHS)


if __name__ == "__main__":
    unittest.main()
