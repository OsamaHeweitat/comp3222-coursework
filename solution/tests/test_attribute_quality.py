"""Add tests for attribute quality checks."""
import numpy as np
import unittest
from solution.attribute_quality import information_gain, information_gain_ratio, chi_squared, chi_squared_yates

class TestAttributeQuality(unittest.TestCase):
    def test_information_gain(self):
        peaty_split = np.array([[4, 0], [1, 5]])
        ig = information_gain(peaty_split)
        self.assertAlmostEqual(ig, 0.610, places=3)

        woody_split = np.array([[2, 3], [3, 2]])
        ig = information_gain(woody_split)
        self.assertAlmostEqual(ig, 0.029, places=3)

        sweet_split = np.array([[2, 5], [3, 0]])
        ig = information_gain(sweet_split)
        self.assertAlmostEqual(ig, 0.396, places=3)

        peaty_woody_split = np.array([[1, 3], [0, 2]])
        ig = information_gain(peaty_woody_split)
        self.assertAlmostEqual(ig, 0.109, places=3)

        peaty_sweet_split = np.array([[0, 5], [1, 0]])
        ig = information_gain(peaty_sweet_split)
        self.assertAlmostEqual(ig, 0.650, places=3)

    def test_information_gain_ratio(self):
        peaty_split = np.array([[4, 0], [1, 5]])
        igr = information_gain_ratio(peaty_split)
        self.assertAlmostEqual(igr, 0.628, places=3)

        woody_split = np.array([[2, 3], [3, 2]])
        igr = information_gain_ratio(woody_split)
        self.assertAlmostEqual(igr, 0.029, places=3)

        sweet_split = np.array([[2, 5], [3, 0]])
        igr = information_gain_ratio(sweet_split)
        self.assertAlmostEqual(igr, 0.449, places=3)

        peaty_woody_split = np.array([[1, 3], [0, 2]])
        igr = information_gain_ratio(peaty_woody_split)
        self.assertAlmostEqual(igr, 0.119, places=3)

        peaty_sweet_split = np.array([[0, 5], [1, 0]])
        igr = information_gain_ratio(peaty_sweet_split)
        self.assertAlmostEqual(igr, 1.000, places=3)

    def test_chi_squared(self):
        peaty_split = np.array([[4, 0], [1, 5]])
        chi2 = chi_squared(peaty_split)
        self.assertAlmostEqual(chi2, 6.667, places=3)

        woody_split = np.array([[2, 3], [3, 2]])
        chi2 = chi_squared(woody_split)
        self.assertAlmostEqual(chi2, 0.400, places=3)

        sweet_split = np.array([[2, 5], [3, 0]])
        chi2 = chi_squared(sweet_split)
        self.assertAlmostEqual(chi2, 4.286, places=3)

        peaty_woody_split = np.array([[1, 3], [0, 2]])
        chi2 = chi_squared(peaty_woody_split)
        self.assertAlmostEqual(chi2, 0.600, places=3)

        peaty_sweet_split = np.array([[0, 5], [1, 0]])
        chi2 = chi_squared(peaty_sweet_split)
        self.assertAlmostEqual(chi2, 6.000, places=3)

    def test_chi_squared_yates(self):
        peaty_split = np.array([[4, 0], [1, 5]])
        chi2y = chi_squared_yates(peaty_split)
        self.assertAlmostEqual(chi2y, 3.750, places=3)

        woody_split = np.array([[2, 3], [3, 2]])
        chi2y = chi_squared_yates(woody_split)
        self.assertAlmostEqual(chi2y, 0.000, places=3)

        sweet_split = np.array([[2, 5], [3, 0]])
        chi2y = chi_squared_yates(sweet_split)
        self.assertAlmostEqual(chi2y, 1.905, places=3)

        peaty_woody_split = np.array([[1, 3], [0, 2]])
        chi2y = chi_squared_yates(peaty_woody_split)
        self.assertAlmostEqual(chi2y, 0.150, places=3)

        peaty_sweet_split = np.array([[0, 5], [1, 0]])
        chi2y = chi_squared_yates(peaty_sweet_split)
        self.assertAlmostEqual(chi2y, 0.960, places=3)

if __name__ == "__main__":
    unittest.main()