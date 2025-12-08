"""Add tests for attribute quality checks."""
import numpy as np
import unittest
from solution.decision_stump import DecisionStumpClassifier

class TestDecisionStump(unittest.TestCase):
    def test_float_in_x(self):
        X = np.array([[1.1, 2], [3, 4]])
        y = np.array([0, 1])
        clf = DecisionStumpClassifier()
        with self.assertRaises(ValueError):
            clf.fit(X, y)
    
    def test_deterministic_attribute_choice_with_random_state(self):
        X = np.array([[1, 2, 3], [3, 4, 5], [1, 4, 6], [3, 2, 7]])
        y = np.array([0, 1, 0, 1])
        clf1 = DecisionStumpClassifier(n_attributes=2, random_state=32)
        clf1.fit(X, y)
        chosen_attr1 = clf1.att_index

        clf2 = DecisionStumpClassifier(n_attributes=2, random_state=32)
        clf2.fit(X, y)
        chosen_attr2 = clf2.att_index

        self.assertEqual(chosen_attr1, chosen_attr2)

        clf1 = DecisionStumpClassifier(n_attributes=2, random_state=53)
        clf1.fit(X, y)
        chosen_attr1 = clf1.att_index

        clf2 = DecisionStumpClassifier(n_attributes=2, random_state=53)
        clf2.fit(X, y)
        chosen_attr2 = clf2.att_index

        self.assertEqual(chosen_attr1, chosen_attr2)
    
    def test_predict_proba_validity(self):
        X = np.array([[1, 2], [3, 4], [1, 4], [3, 2]])
        y = np.array([0, 1, 0, 1])

        clf = DecisionStumpClassifier()
        clf.fit(X, y)
        proba = clf.predict_proba(X)

        self.assertEqual(proba.shape, (len(X), len(np.unique(y))))

        for prob_vector in proba:
            self.assertAlmostEqual(np.sum(prob_vector), 1.0)

        self.assertTrue(np.all(proba >= 0))

    def test_unseen_category_fallback(self):
        X_train = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=int)
        y_train = np.array([0, 0, 1, 1], dtype=int) 

        clf = DecisionStumpClassifier()
        clf.fit(X_train, y_train)

        # tie breaker
        self.assertIsNotNone(clf.att_index)
        self.assertIn(clf.att_index, (0, 1))
        self.assertEqual(clf.att_index, 0)

        X_test = np.array([
            [2, 0],   #unseen
            [2, 1],
        ], dtype=int)

        proba = clf.predict_proba(X_test)

        n_classes = len(np.unique(y_train))
        self.assertEqual(proba.shape, (X_test.shape[0], n_classes))

        root_counts = np.bincount(y_train, minlength=n_classes)
        alpha = clf.alpha                                          
        expected = (root_counts + alpha) / (root_counts.sum() + alpha * n_classes)

        for i in range(X_test.shape[0]):
            self.assertTrue(np.allclose(proba[i], expected),
                            msg=f"proba[{i}] = {proba[i]} != expected {expected}")

        for v in proba:
            self.assertAlmostEqual(np.sum(v), 1.0)


if __name__ == "__main__":
    unittest.main()