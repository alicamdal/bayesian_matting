import unittest
from getNParameters import getNParameters
from getFBParameters import getFBParameters
from getKnownRegions import getKnownRegions
from getMAPestimate import getMAPestimate
import cv2
import numpy as np
import time


class TestBayesMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBayesMethods, self).__init__(*args, **kwargs)
        # reading images for testing
        self.img = cv2.cvtColor(
                                cv2.imread("Samples/unittest_input.png"),
                                cv2.COLOR_BGR2RGB
                                ) / 255
        self.trimap = cv2.imread("Samples/unittest_trimap.png", 0) / 255
        self.gt = cv2.imread("Samples/unittest_gt.png", 0) / 255
        # getting unknown vaues for trimap
        [
            self.f_known,
            self.b_known,
            self.unknown,
            self.alpha
        ] = getKnownRegions(self.img, self.trimap)
        # setting an unknown location
        unk = self.unknown[:, :].astype(np.float64)
        self.X, self.Y = np.nonzero(unk)
        # creating window for testing
        [w, h, c] = self.f_known.shape
        self.x = self.X[2]
        self.y = self.Y[2]
        n1 = int(np.floor(77 / 2))
        n2 = 77 - n1 - 1
        xmin = (np.maximum(0, self.x - n1))
        xmax = (np.minimum(w, self.x + n2)) + 1
        ymin = (np.maximum(0, self.y - n1))
        ymax = (np.minimum(h, self.y + n2)) + 1
        # setting expected values for alpha, foreground, background and ground truth
        trimap = self.trimap.copy()
        self.exp_alpha = trimap[xmin:xmax, ymin:ymax]
        self.exp_alpha[self.exp_alpha == 128/255] = np.float64('nan')
        self.exp_fi = self.f_known[xmin:xmax, ymin:ymax, :]
        self.exp_bi = self.b_known[xmin:xmax, ymin:ymax, :]
        self.exp_gt = self.gt[self.x, self.y]

    def test_getKnownRegions(self):
        # getting function values
        [
            f_known,
            b_known,
            unknown,
            alpha
        ] = getKnownRegions(self.img, self.trimap)
        # gettin mean values for 3rd column
        f = np.mean(f_known, axis=2)
        b = np.mean(b_known, axis=2)
        # clearing all other values except 1 and 0
        f[f != 0] = 1
        b[b != 0] = 0
        # setting unknown location values as 0.5
        unknown[unknown == 1] = 128 / 255
        # setting actual solution
        act_solution = f + b + unknown
        # comparing them with real trimap
        flag = np.array_equal(self.trimap, act_solution)

        self.assertEqual(True, flag)

    def test_getNParameters(self):
        # setting required parameters
        N = 77
        minN = 40
        sigma_g = 10
        # getting function values
        [a, fi, wf, bi, wb] = getNParameters(
                                            self.x,
                                            self.y,
                                            N,
                                            minN,
                                            self.f_known,
                                            self.b_known,
                                            self.alpha,
                                            sigma_g
                                            )
        # setting expected and actual solution for foreground and background
        exp_pixel_solution = [self.exp_fi, self.exp_bi]
        act_pixel_solution = [fi, bi]
        # setting expected and actual solution for alpha
        exp_alpha_solution = [self.exp_alpha]
        act_alpha_solution = [a]
        # comparing them with tolerance
        pixel_flag = np.allclose(
                        exp_pixel_solution,
                        act_pixel_solution,
                        rtol=1e-04,
                        atol=1e-04,
                        equal_nan=True
                        )
        alpha_flag = np.allclose(
                        exp_alpha_solution,
                        act_alpha_solution,
                        rtol=1e-04,
                        atol=1e-04,
                        equal_nan=True
                        )

        self.assertListEqual([True, True], [pixel_flag, alpha_flag])

    def test_getFBParameters(self):
        # creating sample 3D matrix
        known = np.zeros((3, 3, 3))
        # assigning values for computation
        known[:, :, 0] = np.array([
                                [4, 2, 6],
                                [4.2, 2.1, 0.59],
                                [3.9, 2.0, 0.58]
                                ])
        known[:, :, 1] = np.array([
                                [4, 2, 6],
                                [4.2, 2.1, 0.59],
                                [3.9, 2.0, 0.58]
                                ])
        known[:, :, 2] = np.array([
                                [4, 2, 6],
                                [4.2, 2.1, 0.59],
                                [3.9, 2.0, 0.58]
                                ])
        # setting weights as 1
        weights = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        # gettin mean and covariance values for given sample data
        mean, cov = getFBParameters(known, weights)
        # comparing them with tolerance
        mean_flag = np.allclose(
            np.array([[2.8188], [2.8188], [2.8188]]),
            mean,
            rtol=1e-04, atol=1e-04, equal_nan=True
            )
        cov_flag = np.allclose(
            np.array([
                [2.9365, 2.9365, 2.9365],
                [2.9365, 2.9365, 2.9365],
                [2.9365, 2.9365, 2.9365]]),
            cov, rtol=1e-04, atol=1e-04, equal_nan=True
            )

        self.assertListEqual([True, True], [mean_flag, cov_flag])

    def test_getMAPestimate(self):
        # finding neighbourhood parameters for an unknown pixel
        [a, fi, wf, bi, wb] = getNParameters(
                                            self.x,
                                            self.y,
                                            77,
                                            40,
                                            self.f_known,
                                            self.b_known,
                                            self.alpha,
                                            10
                                            )
        # finding foreground and background parameters
        [mean_f, sigma_f] = getFBParameters(fi, wf)
        [mean_b, sigma_b] = getFBParameters(bi, wb)
        # finding initial alpha value
        alpha_init = np.nanmean(a)
        c = np.reshape(self.img[self.x, self.y, :], [3, 1])
        # finding estimated values for f, b and alpha for given pixel
        [f, b, a] = getMAPestimate(
                                mean_f,
                                sigma_f,
                                mean_b,
                                sigma_b,
                                c,
                                0.01,
                                alpha_init,
                                200,
                                1e-6
                                )
        self.assertAlmostEqual(float(self.exp_gt), float(a), places=3)

if __name__ == "__main__":
    unittest.main()
