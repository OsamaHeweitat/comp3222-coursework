import unittest
import numpy as np

from solution.tree_ensemble import TreeEnsembleClassifier


class TestEnsemblePredictProba(unittest.TestCase):

    def setUp(self):
        X_raw = np.array([
            ["yes", "no", "yes"],
            ["yes", "yes", "yes"],
            ["yes", "no", "no"],
            ["yes", "no", "no"],
            ["no",  "yes", "no"],
            ["no",  "yes", "yes"],
            ["no",  "yes", "yes"],
            ["no",  "yes", "yes"],
            ["no",  "no",  "yes"],
            ["no",  "no",  "yes"]
        ])
        y = np.array([
            "Islay", "Islay", "Islay", "Islay", "Islay",
            "Speyside", "Speyside", "Speyside", "Speyside", "Speyside"
        ])

        X = np.zeros_like(X_raw, dtype=int)
        X[X_raw == "yes"] = 1
        X[X_raw == "no"] = 0
        self.X = X
        self.y = y

        self.ensemble = TreeEnsembleClassifier(
            n_estimators=3,
            average_probas=True
        )
        self.ensemble.fit(self.X, self.y)

    def test_predict_proba_shape(self):
        proba = self.ensemble.predict_proba(self.X)
        n_samples = self.X.shape[0]
        n_classes = len(self.ensemble.classes_)

        self.assertEqual(proba.shape, (n_samples, n_classes))

    def test_probabilities_sum_to_one(self):
        proba = self.ensemble.predict_proba(self.X)
        row_sums = np.sum(proba, axis=1)

        self.assertTrue(np.allclose(row_sums, 1.0))

    def test_alignment_works(self):
        """
        Ensures predict_proba still works even if stump classes_ are in
        weird or reversed order.
        """
        for stump, _ in self.ensemble.estimators_:
            stump.classes_ = stump.classes_[::-1]

        proba = self.ensemble.predict_proba(self.X)

        row_sums = np.sum(proba, axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))

    def test_majority_vote_mode(self):
        self.ensemble.average_probas = False
        proba = self.ensemble.predict_proba(self.X)

        self.assertTrue(np.all(np.sum(proba, axis=1) == 1.0))
        self.assertTrue(np.all((proba == 0.0) | (proba == 1.0)))


if __name__ == "__main__":
    unittest.main()
