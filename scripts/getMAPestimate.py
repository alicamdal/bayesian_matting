from numpy import hstack, vstack, square, concatenate, matmul
from numpy import multiply, zeros, sum as npsum, eye, inf
from numpy.linalg import norm
from scipy.linalg import pinv


def getMAPestimate(
                mean_f,
                sigma_f,
                mean_b,
                sigma_b,
                c,
                sigma_c,
                alpha,
                max_iter,
                min_like
                ):
    """
    Function that calculates estimated values 
    for foreground, background and alpha matte

    [inputs = numpy.array, numpy.array, numpy.array, numpy.array, 
                        numpy.array, float, numpy.array, int, float]
        mean_f      => mean values for foreground pixels
        sigma_f     => covariance matrix of foreground pixels
        mean_b      => mean values for background pixels
        sigma_b     => covariance matrix of background pixels
        c = composited image
        sigma_c     => covariance of composited image
        alpha       => initial alpha matte
        max_iter    => limit of maximum iterations
        min_like    => minimum value of likelihood for estimated values
    
    [outputs = numpy.array, numpy.array, numpy.array]
        f           => estimated values of given foreground
        b           => estimated values of given background
        alpha       => estimated values of given alpha matte
    """
    # creating identity matrix
    I = eye(3)
    # initial likelihood
    lastlike = -inf
    # inversion of covariance matrices
    invSigma_f = pinv(sigma_f)
    invSigma_b = pinv(sigma_b)
    # squared sigma c calculation for multiple use
    squared_sigma_c = square(sigma_c)
    # iteration begins
    for i in range(max_iter):
        """
        This iteration solves A.x = b linear equation in the first hand then
        with estimated values it solves alpha equation
        """
        # calculating A
        A_indx_11 = invSigma_f + (I * (square(alpha) / (squared_sigma_c)))
        A_indx_12 = (I * (alpha * (1 - alpha)) / (squared_sigma_c))
        A_indx_22 = invSigma_b + I * (square((1 - alpha) / sigma_c))
        row_1 = hstack((A_indx_11, A_indx_12))
        row_2 = hstack((A_indx_12, A_indx_22))
        A = vstack((row_1, row_2))
        # calculating b
        b_indx_11 = matmul(invSigma_f, mean_f) + (c * alpha) / (squared_sigma_c)
        b_indx_12 = matmul(invSigma_b, mean_b) + (c * (1-alpha)) / (squared_sigma_c)
        b = concatenate((b_indx_11, b_indx_12))
        # solving equation for finding estimated foreground and background values
        x = matmul(pinv(A), b)
        # finding foreground and background
        f = x[0:3]
        b = x[3:]
        # solving for alpha with estimated foreground and background values
        alpha = multiply((c - b), (f - b)).sum(axis=0) / square(norm((f - b)))
        alpha = max(0, min(1, alpha))
        # calculating likelihood of estimated values
        Lf = - matmul(matmul((f - mean_f).T, invSigma_f), (f - mean_f)) / 2
        Lb = - matmul(matmul((b - mean_b).T, invSigma_b), (b - mean_b)) / 2
        La = - (square(
                    norm(c - (alpha * f) - ((1 - alpha) * b)))
                ) / (squared_sigma_c)
        # finding summation of log likelihood
        loglike = Lf + Lb + La
        # deciding continue or break
        if abs(loglike - lastlike) <= min_like:
            break
        # storing current likelihood
        lastlike = loglike

    return f, b, alpha
