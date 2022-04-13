from scripts.bayesAPI import bayesMat
from scripts.bayesmat import main
import argparse
import cv2
import time
import numpy as np
import math
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--trimap", type=str, default="")
    parser.add_argument("--gt", type=str, default="")
    parser.add_argument("--new-bg", type=str, default="")
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


def createComposite(bg_inp, fg_inp, alpha):
    new_bg = cv2.imread(bg_inp) / 255
    fg_inp = cv2.imread(fg_inp) / 255
    alpha_cp = alpha
    if new_bg.shape[0] != fg_inp.shape[0] or new_bg.shape[1] != fg_inp.shape[1]:
        new_shp_bg = Image.new("RGB", (fg_inp.shape[1], fg_inp.shape[0]))
        new_shp_bg.paste(Image.open(bg_inp), (0, 0))
        new_bg = np.array(new_shp_bg)
        new_bg = cv2.cvtColor(new_bg, cv2.COLOR_RGB2BGR)
        new_bg = new_bg / 255
        print("""
            ERROR
            New Background width or height is different than input's.
            """)

    if alpha_cp.shape[-1] != 3:
        new_alpha = Image.new("RGB", (alpha_cp.shape[1], alpha_cp.shape[0]))
        new_alpha.paste(Image.fromarray(alpha_cp * 255), (0,0))
        alpha_cp = np.array(new_alpha)
        alpha_cp[alpha_cp > 155] = 255
        alpha_cp[alpha_cp != 255] = 0
        alpha_cp = cv2.cvtColor(alpha_cp, cv2.COLOR_RGB2BGR) / 255

    alpha_mask = alpha_cp.copy()
    alpha_mask[alpha_cp != 0] = 0
    alpha_mask[alpha_cp == 0] = 1
    bg = alpha_mask * new_bg
    fg = alpha_cp * fg_inp
    result = np.hstack((alpha_cp, bg + fg))
    cv2.imshow("results", result)
    cv2.imwrite("outputs/output_alpha.png", alpha_cp * 255)
    cv2.imwrite("outputs/output_foreground.png", fg * 255)
    cv2.imwrite("outputs/output_background.png", bg * 255)
    cv2.imwrite("outputs/output_composited.png", (bg + fg) * 255)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()

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
    
    createComposite(args.new_bg, args.input, alpha)