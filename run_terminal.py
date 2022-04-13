from scripts.bayesAPI import bayesMat
from scripts.bayesmat import main
import argparse
import cv2
import time
import numpy as np
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--trimap", type=str, default="")
    parser.add_argument("--gt", type=str, default="")
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--sigmac", type=float, default=0.01)
    parser.add_argument("--sigmag", type=int, default=10)
    parser.add_argument("--minN", type=int, default=40)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--min-like", type=float, default=1e-6)
    parser.add_argument("--cpu-count", type=int, default=4)
    parser.add_argument("--output-name", type=str, default=None)
    args = parser.parse_args()
    return args


def calculateMetrics(gt_img, alpha, selector):
        if selector == 0:
            gt_img = cv2.imread(gt_img) / 255
            mse = np.mean((gt_img - alpha) ** 2)
            psnr = (
                    20 * math.log10(1.0 / math.sqrt(mse))
                    )
            psnr = "%.2f" % psnr
            mse = "%.5f" % mse
            return psnr, mse
        else:
            gt_img = cv2.imread(gt_img, 0) / 255
            mse = np.mean((gt_img - alpha) ** 2)
            psnr = (
                    20 * math.log10(1.0 / math.sqrt(mse))
                    )
            psnr = "%.2f" % psnr
            mse = "%.5f" % mse
            return psnr, mse


if __name__ == "__main__":
    args = parse_args()
    if args.cpu_count == 1:
        start = time.time()
        alpha = main(
                    args.input,
                    args.trimap,
                    args.window_size,
                    args.sigmac,
                    args.sigmag,
                    args.minN,
                    args.max_iter,
                    args.min_like,
                    True
                    )
        calc_time = time.time() - start
        psnr, mse = calculateMetrics(args.gt, alpha, selector=1)
        print(f"Process took : {int(calc_time)} seconds")
        print(f"PSNR : {psnr} dB")
        print(f"MSE : {mse}")

    else:
        bayes = bayesMat(
                        input=args.input,
                        trimap=args.trimap,
                        N=args.window_size,
                        sigma_c=args.sigmac,
                        sigma_g=args.sigmag,
                        minN=args.minN,
                        max_iter=args.max_iter,
                        min_like=args.min_like,
                        progress_bar=True,
                        cpu_count=args.cpu_count,
                        output_name=args.output_name
                        )
        alpha, calc_time = bayes.run()
        psnr, mse = calculateMetrics(args.gt, alpha, selector=0)
        print(f"Process took : {int(calc_time)} seconds")
        print(f"PSNR : {psnr} dB")
        print(f"MSE : {mse}")
