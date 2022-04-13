from scripts.bayesAPI import bayesMat
from skimage.metrics import structural_similarity
from guppy import hpy
import networkx as nx
import argparse
import numpy as np
import math as math
import cv2
import os
import multiprocessing

class e2eTest:
    def __init__(self):
        # initialize class variables
        self.path = os.getcwd()
        self.iter = []
        self.wind = []
        self.neigh = []
        self.minL = []
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.time = []
        self.mem = []

    def loadData(self, i):
        # read groundtruth from the directory
        gt_img = cv2.imread(self.path + "\\scripts\\Samples\\gt" + str(i) +
                            ".png") / 255
        # set path for input image
        input = self.path + "\\scripts\\Samples\\input" + str(i) + ".png"
        # set path for trimap image
        trimap = self.path + "\\scripts\\Samples\\trimap" + str(i) + ".png"
        return gt_img, input, trimap

    def setBayesParams(self, param, num_params):
        # set dafault matting parameters
        self.iter = [200] * num_params
        self.wind = [25] * num_params
        self.neigh = [40] * num_params
        self.minL = [1e-6] * num_params
        # adjust the bayesian parameters for matting analysis
        if param == 'iter':
            # adjusting the max iterations
            self.iter.clear()
            self.iter = [75, 100, 150, 175, 200]
        elif param == 'wind':
            # adjusting the window size
            self.wind.clear()
            self.wind = [15, 25, 35, 45, 55]
        elif param == 'neigh':
            # adjusting the min required neighbourhood pixels
            self.neigh.clear()
            self.neigh = [10, 20, 30, 40, 50]
        elif param == 'minL':
            # adjusting the min likelihood values
            self.minL.clear()
            self.minL = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    def bayesE2e(self, i, input, trimap, output_file):
        # invoke the bayesian matting for current inputs
        bayes = bayesMat(
                    input=input,
                    trimap=trimap,
                    N=self.wind[i],
                    sigma_c=0.01,
                    sigma_g=10,
                    minN=self.neigh[i],
                    max_iter=self.iter[i],
                    min_like=self.minL[i],
                    progress_bar=True,
                    cpu_count = multiprocessing.cpu_count(),
                    output_name=output_file
                )
        alpha, cp_time = bayes.run()
        #alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
        return alpha, cp_time

    def calcComputLoad(self):
        # calculate computational load
        h = hpy()
        num_nodes = 1000
        num_edges = 5000
        G = nx.gnm_random_graph(num_nodes, num_edges)
        x = h.heap()
        self.mem.append(x.size)
        return x

    def calcMetrics(self, n, param, num_params):
        for i in range(1, n + 1):
            # load images
            gt_img, input, trimap = self.loadData(i)
            # set matting params
            self.setBayesParams(param, num_params)
            # calculate alpha for each num_params
            for j in range(num_params):
                # set output filename
                output_file = "output" + str(i) + str(j) + ".png"
                # get alpha for each matt parameters
                alpha, cp_time = self.bayesE2e(j, input, trimap, output_file)
                # write loss as an image
                #cv2.imwrite(self.path + "\\scripts\\Samples\\loss" + str(i) +
                            #".png", gt_img - alpha)
                # calculate MSE for alpha and ground truth
                #alpha = alpha / 255
                self.mse.append(np.mean((gt_img - alpha) ** 2))
                if self.mse[j] == 0:
                    return 100
                pixel_max = 1.0
                # calculate psnr
                self.psnr.append(20 * math.log10(pixel_max /
                                 math.sqrt(self.mse[j])))
                # calculate ssim
                #(score, diff) = 
                score = structural_similarity(gt_img, alpha, multichannel=True)
                #diff = (diff * 255).astype("uint8")
                self.ssim.append(score)
                self.time.append(cp_time)
                x = self.calcComputLoad()
                if i == 1:
                    print(x)
            print('MSE:', self.mse)
            self.mse.clear()
            print('PSNR:', self.psnr)
            self.psnr.clear()
            print('SSIM:', self.ssim)
            self.ssim.clear()
            print('Time in sec:', self.time)
            self.time.clear()
            print('Memory in Bytes:', self.mem)
            self.mem.clear()

    def getMetrics(self, n, param, num_params):
        self.calcMetrics(n, param, num_params)


def parse_opt():
    parser = argparse.ArgumentParser(description='end to end testing for \
                                     Bayesian Matting')
    parser.add_argument('--numImages', type=int, default=5, help='number of \
                        images to be processed')
    parser.add_argument('--paramVariation', type=str, default='', help='values:\
                        "", "iter", "wind", "neigh", "minL"')
    parser.add_argument('--numVariations', type=int, default=1, help='number \
                        of variations to be processed')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    obj1 = e2eTest()
    # get metrics for images with default values
    obj1.getMetrics(opt.numImages,
                    opt.paramVariation,
                    opt.numVariations)
