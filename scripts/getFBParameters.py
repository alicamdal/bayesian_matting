import numpy as np
np.seterr(divide='ignore', invalid='ignore')

def getFBParameters(known, weights):
    """
    Function that calculates foreground and background parameters.

    [inputs = numpy.array, numpy.array]
        known   => known pixel value of foreground or background
        weights => weights of known pixels 
    
    [outputs = numpy.array, numpy.array]
        mean    => mean values of given known pixels
        sigma   => covariance matrix of given known pixels
    """
    # finding weight's shape values
    weight_shapes = weights.shape
    # converting NaN values to 0 in weight matrix
    weights[np.isnan(weights)] = 0
    # finding summation of weights
    W = np.sum(weights)
    # finding mean values of given known pixels
    mean = np.array([
        [np.sum(weights * known[:, :, 0])],
        [np.sum(weights * known[:, :, 1])],
        [np.sum(weights * known[:, :, 2])]]) / W
    # finding covariance matrix of given known pixels
    sigma = np.zeros((3, 3))
    for i in range(0, weight_shapes[0]):
        for j in range(0, weight_shapes[1]):
            if weights[i, j] != 0:
                pixel = np.array([
                                [
                                    known[i, j, 0],
                                    known[i, j, 1],
                                    known[i, j, 2]
                                ]
                                ]).T
                sigma = sigma + np.multiply(
                                            weights[i, j],
                                            np.matmul(
                                                (pixel - mean),
                                                (pixel - mean).T
                                                )
                                            )
    sigma = sigma / W
    return mean, sigma
