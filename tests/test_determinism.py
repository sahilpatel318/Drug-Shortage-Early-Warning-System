import unittest

from ews import pipeline, config


class TestDeterminism(unittest.TestCase):
    def test_two_runs_identical(self):
        r1 = pipeline.run(seed=config.SEED)
        r2 = pipeline.run(seed=config.SEED)
        self.assertEqual(r1["determinism_hash"], r2["determinism_hash"])

    def test_seed_changes_output(self):
        r1 = pipeline.run(seed=config.SEED)
        r2 = pipeline.run(seed=config.SEED + 1)
        self.assertNotEqual(r1["determinism_hash"], r2["determinism_hash"])


if __name__ == "__main__":
    unittest.main()
