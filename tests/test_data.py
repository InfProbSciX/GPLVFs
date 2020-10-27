
import sys
import unittest
import numpy as np

sys.path.append('/Users/adityaravuri/Documents/GitHub/GPLVFs/')
from gplvf import data


class RealDataTests(unittest.TestCase):
    cases = ('iris', 'oilflow', 'gene', 'mnist')

    def setUp(self):
        self.data = {case: data.load_real_data(case) for case in self.cases}

    def test_shapes(self):
        for case in self.cases:
            n, d, q, X, Y, labels = self.data[case]
            self.assertEqual(d, len(Y.T))
            self.assertEqual(n, len(Y))
            self.assertEqual(n, len(labels))

    def test_invalid_dataset(self):
        with self.assertRaises(NotImplementedError):
            data.load_real_data('')


class SyntheticLatentTests(unittest.TestCase):
    labels = ('blobs', 'noisy_circles', 'make_moons', 'varied', 'normal')

    def setUp(self, n=42):
        self.n = n
        self.latent_data =\
            {lb: data._load_2d_synthetic_latent(lb, n) for lb in self.labels}

    def test_shapes(self):
        for lb in self.latent_data:
            X, labels = self.latent_data[lb]
            self.assertEqual(self.n, len(X))
            self.assertEqual(2, len(X.T))
            self.assertEqual(self.n, len(labels))

    def test_invalid_dataset(self):
        with self.assertRaises(NotImplementedError):
            data._load_2d_synthetic_latent('')

    def test_potential_three(self):
        X = data._load_2d_weird_latent(self.n)
        self.assertEqual(self.n, len(X))
        self.assertEqual(2, len(X.T))


class SyntheticDataTests(unittest.TestCase):
    def setUp(self, n=42):
        self.cases = [
            data.generate_synthetic_data(n),
            data.generate_synthetic_data(n, y_type='lo_dim'),
            data.generate_synthetic_data(n, 'blobs', 'hi_dim'),
            data.generate_synthetic_data(n, 'noisy_circles', 'lo_dim'),
            data.generate_synthetic_data(n, 'make_moons', 'hi_dim'),
            data.generate_synthetic_data(n, 'normal', 'by_cat'),
            data.generate_synthetic_data(n, 'varied', 'lo_dim')
        ]

    def test_inits(self):
        for (n, d, q, X, Y, _) in self.cases:
            self.assertEqual(d, len(Y.T))
            self.assertEqual(n, len(Y))
            self.assertEqual(n, len(X))
            self.assertEqual(q, len(X.T))
            self.assertFalse(np.isnan(X).any())
            self.assertFalse(np.isnan(Y).any())


class PlotTests(unittest.TestCase):
    def setUp(self):
        _, _, _, X, Y, _ = data.generate_synthetic_data(10)
        self._XY = X, Y

    def test_plot(self):
        X, Y = self._XY
        data.check_model(X, Y, Y)


if __name__ == '__main__':
    unittest.main()
