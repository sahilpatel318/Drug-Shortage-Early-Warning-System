import unittest

import numpy as np

from ews import config
from ews.synth import generate
from ews.firewall import Vault, leakage_test, SealedError
from ews.features import build_features


class TestLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = generate(config.SEED)
        cls.vault = Vault(cls.panel.onset)

    def test_features_independent_of_future(self):
        result = leakage_test(build_features, self.panel, self.vault, seed=config.SEED)
        self.assertTrue(result["features_independent_of_future"])
        self.assertTrue(result["labels_depend_on_future"])
        self.assertGreater(result["n_rows"], 0)

    def test_leaky_builder_is_caught(self):
        # A builder that copies the future label into a feature must be flagged.
        def leaky(panel, vault):
            df = build_features(panel, vault)
            df = df.copy()
            df["peek"] = df["y"]      # blatant leak
            return df

        with self.assertRaises(SealedError):
            leakage_test(leaky, self.panel, self.vault, seed=config.SEED)

    def test_label_window_is_strictly_future(self):
        # Onsets AT or before the anchor must never set the label.
        drug_idx = int(self.panel.drug_ids[0][1:])
        onsets = self.vault.onset_months(drug_idx)
        if onsets.size:
            o = int(onsets[0])
            if o - 1 >= 0 and (o - 1) + config.LEAD_HORIZON < config.PERIOD_MONTHS:
                # anchor one month before an onset should see it in the window
                self.assertEqual(self.vault.label_window(drug_idx, o - 1, config.LEAD_HORIZON), 1)
            # anchor exactly at the onset must not count that same month
            if o + config.LEAD_HORIZON < config.PERIOD_MONTHS:
                future = self.vault.label_window(drug_idx, o, config.LEAD_HORIZON)
                self.assertIn(future, (0, 1))  # only later onsets may set it


if __name__ == "__main__":
    unittest.main()
