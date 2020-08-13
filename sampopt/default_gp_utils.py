from math import pi

import numpy as np


def sq_exp_cov(var, rho, x=None, y=None, d=None):
    """ Computes the exponential covariance function
    Note that this is a special case of the Matern covariance function with
    nu=0.5

    Args:
        var (float): positive value that scales the variance, the smaller
            this quantity the stronger the spatial dependence.
        rho (float): positive value that scales the distance, the larger
            this quantity the stronger the spatial dependence.
        x (narray): location of the first set of data
        y (narray): location of the second set of data
        d (narray): Optional, this should be the same dimension as
            (len(x), len(y))

    Returns:
        cov_matrix (narray): A covariance matrix where the i-th row
            and j-th column represents the covariance between x[i] and y[j]
    """
    if d is None:
        d = dist_matrix(x, y)
    return var * np.exp(-np.power(d, 2) / rho)


def exp_cov(var, rho, x=None, y=None, d=None):
    """ Computes the exponential covariance function
    Note that this is a special case of the Matern covariance function with
    nu=0.5

    Args:
        var (float): positive value that scales the variance, the smaller
            this quantity the stronger the spatial dependence.
        rho (float): positive value that scales the distance, the larger
            this quantity the stronger the spatial dependence.
        x (narray): location of the first set of data
        y (narray): location of the second set of data
        d (narray): Optional, this should be the same dimension as
            (len(x), len(y))

    Returns:
        cov_matrix (narray): A covariance matrix where the i-th row
            and j-th column represents the covariance between x[i] and y[j]
    """
    if d is None:
        d = dist_matrix(x, y)
    return var * np.exp(-d / rho)


def dist_matrix(x0, x1,
                pos_fun=lambda x: np.power(x, 2),
                rescale_fun=lambda x: np.power(x, 1/2)):
    if len(x0.shape) < 2:
        x0 = x0.reshape(-1, 1)
    if len(x1.shape) < 2:
        x1 = x1.reshape(-1, 1)
    x0n, x0p = x0.shape
    x1n, x1p = x1.shape
    dist = np.zeros((x0n, x1n))

    assert x0p == x1p

    # Calculate the distance between each dimension separately
    # using broadcasting from numpy
    for i in range(x0p):
        dist += pos_fun(x0[:, i].reshape(-1, 1)
                        - x1[:, i].reshape(1, -1))

    return rescale_fun(dist)


def neg_log_likelihood_gp(params, d, y, cov_fun):
    var = params[0]
    rho = params[1]
    mu = params[2]
    noise_var = params[3]
    if var < 0 or rho < 0 or noise_var < 0:
        return np.inf
    # TODO, cov_fun should be more general
    cov_mat = cov_fun(var, rho, d=d) + np.diag(noise_var * np.ones(len(y)))
    sign, logdet = np.linalg.slogdet(cov_mat)
    assert sign > 0

    neg_log_likelihood = (
        0.5 * (y - mu).T.dot(np.linalg.inv(cov_mat)).dot(y - mu)
        + (logdet / 2 + cov_mat.shape[0] / 2 * np.log(2 * pi)))
    if isinstance(neg_log_likelihood, float):
        out = neg_log_likelihood
    else:
        out = neg_log_likelihood.trace()

    return out
