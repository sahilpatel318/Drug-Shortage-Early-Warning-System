import unittest

from ews import config, pipeline


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = pipeline.run(seed=config.SEED)

    def test_lr_beats_prevalence(self):
        pr = self.report["pr_auc"]
        self.assertGreater(pr["logistic_regression"], pr["prevalence"])

    def test_lr_beats_random(self):
        pr = self.report["pr_auc"]
        self.assertGreater(pr["logistic_regression"], pr["random"])

    def test_lr_beats_best_single_feature(self):
        pr = self.report["pr_auc"]
        self.assertGreaterEqual(pr["logistic_regression"], pr["best_single_feature"])

    def test_threshold_metrics_valid(self):
        clf = self.report["classification_at_threshold"]
        for key in ("precision", "recall", "f1"):
            self.assertGreaterEqual(clf[key], 0.0)
            self.assertLessEqual(clf[key], 1.0)
        self.assertEqual(clf["tp"] + clf["fn"], clf["positives"])

    def test_firewall_passed(self):
        fw = self.report["firewall"]
        self.assertTrue(fw["features_independent_of_future"])
        self.assertTrue(fw["labels_depend_on_future"])

    def test_lead_time_bounded_by_window(self):
        lead = self.report["lead_time"]
        if lead["median_lead_days"] is not None:
            self.assertLessEqual(lead["max_lead_days"],
                                 config.WARN_LOOKBACK * 31 + 1)


if __name__ == "__main__":
    unittest.main()
