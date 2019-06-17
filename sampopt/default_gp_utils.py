from math import pi

import numpy as np


def exp_cov(var, rho, x=None, y=None, d=None):
    if d is None:
        d = dist_matrix(x, y)
    return var * np.exp(-d / rho)


def dist_matrix(x0, x1,
                pos_fun=lambda x: np.power(x, 2),
                rescale_fun=lambda x: np.power(x, 1/2)):
    x0n, x0p = x0.shape
    x1n, x1p = x1.shape
    dist = np.zeros((x0n, x1n))

    assert x0p == x1p

    for i in range(x0p):
        dist += pos_fun(x0[:, i].reshape(-1, 1) - x1[:, i].reshape(1, -1))

    return rescale_fun(dist)


def neg_log_likelihood_exp_gp(params, d, y):
    var = params[0]
    rho = params[1]
    mu = params[2]
    if var < 0 or rho < 0:
        return np.inf
    cov_mat = exp_cov(var, rho, d=d)
    sign, logdet = np.linalg.slogdet(cov_mat)
    assert sign > 0

    neg_log_likelihood = (
        0.5 * np.matmul(np.matmul((y - mu).T, cov_mat),
                        y - mu)
        + cov_mat.shape[0] / 2 * (logdet + np.log(2 * pi))
    )

    return neg_log_likelihood.trace()
