import sampopt.default_gp_utils as utils
import numpy as np


x = np.linspace(0, 1, 11)
y = x[:6]

var_par = 0.5
rho_par = 0.3

d_mat = utils.dist_matrix(x, y)


def test_sq_exp_cov():
    cov_mat = utils.sq_exp_cov(var_par, rho_par, x=x, y=y)
    assert cov_mat[0, 0] == var_par
    cov_sub = cov_mat[:len(y), :len(y)]
    assert np.mean(np.abs(cov_sub.T - cov_sub)) < 1e-12
    decreasing = [cov_mat[0, i] > cov_mat[0, i+1] for i in range(len(y) - 1)]
    assert all(decreasing)


def test_exp_cov():
    cov_mat = utils.exp_cov(var_par, rho_par, x=x, y=y)
    assert cov_mat[0, 0] == var_par
    cov_sub = cov_mat[:len(y), :len(y)]
    assert np.mean(np.abs(cov_sub.T - cov_sub)) < 1e-12
    assert np.abs(cov_mat[0, 3] - var_par * np.exp(-1)) < 1e-12
    decreasing = [cov_mat[0, i] > cov_mat[0, i+1] for i in range(len(y) - 1)]
    assert all(decreasing)


def test_dist_matrix():
    assert d_mat.shape == (len(x), len(y))
    self_zero_dist = [d_mat[i, i] == 0 for i in range(len(y))]
    assert all(self_zero_dist)
    incr_dist = [d_mat[0, i] < d_mat[0, i+1] for i in range(len(y) - 1)]
    assert all(incr_dist)


def test_neg_log_likelihood_gp():
    d_mat = utils.dist_matrix(y, y)
    data = np.zeros_like(y)
    nll_0 = utils.neg_log_likelihood_gp([var_par, rho_par, 0, 0],
                                        d_mat, data,
                                        utils.exp_cov)
    data = np.ones_like(y)
    nll_1 = utils.neg_log_likelihood_gp([var_par, rho_par, 1, 0],
                                        d_mat, data,
                                        utils.exp_cov)
    assert nll_0 == nll_1
    nll_10 = utils.neg_log_likelihood_gp([var_par, rho_par, 0, 0],
                                         d_mat, data,
                                         utils.exp_cov)
    assert nll_1 < nll_10
