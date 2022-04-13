from cv2 import cvtColor, COLOR_BGR2RGB
from cv2 import imread, imshow, waitKey, destroyAllWindows
from numpy import nanmean, nonzero, reshape, float64
from scripts.getNParameters import getNParameters
from scripts.getFBParameters import getFBParameters
from scripts.getKnownRegions import getKnownRegions
from scripts.getMAPestimate import getMAPestimate
import numpy as np


def getIndex(img):
    # Finding Index of given Input Image
    data = img.split("/")
    data = data[-1]
    data = data.split(".")
    if len(data[0]) == 7:
        indx = data[0][-1]
    elif len(data[0]) == 8:
        indx = data[0][-2] + data[0][-1]
    return indx


def progressBar(
        iteration,
        total,
        prefix='',
        suffix='',
        decimals=1,
        length=100,
        fill='█',
        printEnd="\r"
        ):
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if iteration == total:
        print()


def main(
        input_img,
        trimap,
        N,
        sigma_c,
        sigma_g,
        minN,
        max_iter,
        min_like,
        progress_flag
        ):
    if not progress_flag:
        # Main Function of Bayesian Matting
        # finding index of given input
        indx = getIndex(input_img)
        # reading input and trimap
        img = cvtColor(imread(input_img), COLOR_BGR2RGB) / 255
        trimap = imread(trimap, 0) / 255
        # finding known regions of given images
        [f_known, b_known, unk, alpha] = getKnownRegions(img, trimap)
        # finding unknown pixel number and locations
        unk = unk[:, :].astype(float64)
        X, Y = nonzero(unk)
        # iterating over unknown pixels
        for i in range(len(Y)):
            x = X[i]
            y = Y[i]
            # finding neighbourhood parameters for an unknown pixel
            [a, fi, wf, bi, wb] = getNParameters(
                                                x,
                                                y,
                                                N,
                                                minN,
                                                f_known,
                                                b_known,
                                                alpha,
                                                sigma_g
                                                )
            # finding foreground and background parameters
            [mean_f, sigma_f] = getFBParameters(fi, wf)
            [mean_b, sigma_b] = getFBParameters(bi, wb)

            # finding initial alpha value
            alpha_init = nanmean(a)
            c = reshape(img[x, y, :], [3, 1])

            if not np.isnan(np.sum(sigma_b)) and not np.isnan(np.sum(sigma_f)):
                # finding estimated values for f, b and alpha for given pixel
                [f, b, a] = getMAPestimate(
                                        mean_f,
                                        sigma_f,
                                        mean_b,
                                        sigma_b,
                                        c,
                                        sigma_c,
                                        alpha_init,
                                        max_iter,
                                        min_like
                                        )
                # updating  known pixels and alpha
                f_known[x, y, :] = f.ravel()
                b_known[x, y, :] = b.ravel()
                alpha[x, y] = a
            elif not np.isnan(np.sum(sigma_b)) and np.isnan(np.sum(sigma_f)):
                alpha[x, y] = 0
            elif np.isnan(np.sum(sigma_b)) and not np.isnan(np.sum(sigma_f)):
                alpha[x, y] = 1

        return [indx, alpha]

    else:
        # reading input and trimap
        img = cvtColor(imread(input_img), COLOR_BGR2RGB) / 255
        trimap = imread(trimap, 0) / 255
        # finding known regions of given images
        [f_known, b_known, unk, alpha] = getKnownRegions(img, trimap)
        # finding unknown pixel number and locations
        unk = unk[:, :].astype(float64)
        X, Y = nonzero(unk)
        # iterating over unknown pixels
        progressBar(
            0,
            len(Y),
            prefix="Progress:", suffix="Complete", length=100)

        for i in range(len(Y)):
            x = X[i]
            y = Y[i]
            # finding neighbourhood parameters for an unknown pixel
            [a, fi, wf, bi, wb] = getNParameters(
                                                x,
                                                y,
                                                N,
                                                minN,
                                                f_known,
                                                b_known,
                                                alpha,
                                                sigma_g
                                                )
            # finding foreground and background parameters
            [mean_f, sigma_f] = getFBParameters(fi, wf)
            [mean_b, sigma_b] = getFBParameters(bi, wb)

            # finding initial alpha value
            alpha_init = nanmean(a)
            c = reshape(img[x, y, :], [3, 1])

            # finding estimated values for f, b and alpha for given pixel
            [f, b, a] = getMAPestimate(
                                    mean_f,
                                    sigma_f,
                                    mean_b,
                                    sigma_b,
                                    c,
                                    sigma_c,
                                    alpha_init,
                                    max_iter,
                                    min_like
                                    )
            # updating  known pixels and alpha
            f_known[x, y, :] = f.ravel()
            b_known[x, y, :] = b.ravel()
            alpha[x, y] = a

            progressBar(
                    i,
                    len(Y),
                    prefix="Progress:",
                    suffix="Complete", length=100)

        return alpha
