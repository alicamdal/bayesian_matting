import numpy as np


def getKnownRegions(img, trimap):
    """
    Function that can find known regions for bayesian matting.
    
    [Inputs = numpy.array, numpy.array]
        img     => input image
        trimap  => trimatte of input image
    
    [outputs = numpy.array, numpy.array, numpy.array, numpy.array]
        f_known => known pixels of foreground
        b_known => known pixels of background
        unMask  => unknown pixels
        alpha   => initial alpha matte  
    """
    # value of 0.5 for unknown pixels
    #mid = 0.4
    # finding known foreground
    fgMask1 = trimap == 1
    f_known = img.copy()
    fgMask = fgMask1 == False
    f_known[fgMask] = 0
    # finding known background
    bgMask = trimap == 0
    b_known = img.copy()
    bgMask = bgMask == False
    b_known[bgMask] = 0
    # finding unknown pixels and converting 0.5 values to 1
    unMask = trimap.copy()
    unMask[trimap == 1] = 0
    unMask[unMask != 0] = 1
    # finding initial alpha with NaN values
    alpha = np.zeros(trimap.shape)
    alpha[fgMask1] = 1
    alpha[unMask == 1] = np.float64('nan')

    return f_known, b_known, unMask, alpha
