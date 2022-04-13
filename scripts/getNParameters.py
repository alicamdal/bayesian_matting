import numpy as np
import cv2


def getNParameters(x, y, N, minN, f_known, b_known, alpha, sigma_g):
    """
    Function that can find neighbourhood parameters of given unknown pixel.
    
    [inputs = int, int, int, int, numpy.array, numpy.array, numpy.array, int]
        x       => x location of unknown pixel
        y       => y location of unknown pixel
        N       => initial window size
        minN    => minimum number for known pixels of background and foreground
        f_known => known pixels for foreground
        b_known => known pixels for background
        alpha   => alpha matte of given data
        sigma_g => gaussian fall off parameter
    
    [outputs = numpy.array, numpy.array, numpy.array, numpy.array, numpy.array]
        a       => windowed array of alpha matte
        fi      => windowed array of known foreground
        wf      => weights of given windowed foreground
        bi      => windowed array of known background
        wb      => weights of given windowed background
    """
    for i in range(400):
        # find width, height and channel value for windowing
        [w, h, c] = f_known.shape
        # calculating window values
        n1 = int(np.floor(N / 2))
        n2 = N - n1 - 1
        # finding x and y locations of window
        xmin = (np.maximum(0, x - n1))
        xmax = (np.minimum(w, x + n2)) + 1
        ymin = (np.maximum(0, y - n1))
        ymax = (np.minimum(h, y + n2)) + 1
        # finding windowed arrays
        a = alpha[xmin:xmax, ymin:ymax]
        fi = f_known[xmin:xmax, ymin:ymax, :]
        bi = b_known[xmin:xmax, ymin:ymax, :]
        # calculating gaussian kernel
        gi = getGaussianKernel(N, sigma_g)
        gi = gi / np.amax(gi[:])
        # calculating gaussian fall off
        fall_off = calcGaussian(x, y, gi, alpha, a)
        # finding weights of given known pixels
        wf = np.square(a) * fall_off
        wb = np.square(1 - a) * fall_off
        # finding known pixel weight counts
        nwf = wf[wf > 0]
        nwb = wb[wb > 0]
        # deciding continue or breaking
        if len(nwb) < minN or len(nwf) < minN:
            N = N + 4
            continue
        else:
            break
    return a, fi, wf, bi, wb


def getGaussianKernel(size, sigma):
    """
    Function that calculates gaussian weight.
    
    [inputs = int, int]
        size    => window size
        sigma   => sigma value for gaussian kernel
    
    [output = numpy.array]
        kernel  => gaussian kernel for given data
    """ 
    # finding 1d gaussian kernel with opencv
    gauss_temp = cv2.getGaussianKernel(size, sigma)
    # converting 1d gaussian kernel to 2d
    kernel = np.dot(gauss_temp, gauss_temp.T)
    return kernel


def calcGaussian(row, col, gi, alpha, wind_alpha):
    """
    Function that calculates gaussian fall off.
    
    [inputs = int, int, numpy.array, numpy.array, numpy.array]
        row         => x location of given unknown pixel
        col         => y location of given unknown pixel
        gi          => gaussian kernel
        alpha       => alpha matte of given data
        wind_alpha  => windowed alpha of given data
    
    [output = numpy.array]
        gi          => calculated gaussian fall off
    """
    # finding shape values of given data
    [gi_r, gi_c] = gi.shape
    [alpha_r, alpha_c] = alpha.shape
    [a_r, a_c] = wind_alpha.shape
    [Nr, Nc] = gi.shape
    # if blocks for modifying gaussian fall off
    if (row + 1) <= np.floor(gi_r / 2):
        gi = np.delete(gi, np.s_[0:((gi_r - a_r))], axis=0)
    if (row + 1) >= (alpha_r - np.floor(gi_r / 2)):
        gi = gi[0:Nr - ((gi_r - a_r)), :]
    if (col + 1) <= np.floor(gi_c / 2):
        gi = np.delete(gi, np.s_[0:((gi_c - a_c))], axis=1)
    if (col + 1) >= alpha_c - np.floor(gi_c / 2):
        gi = gi[:, 0:Nc - ((gi_c - a_c))]

    return gi
