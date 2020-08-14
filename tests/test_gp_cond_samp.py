import sampopt.default_gp_utils as utils
import sampopt.gp_cond_samp as gp_utils
import numpy as np

n = 10
p = 3


def y_fun(x):
    return np.sin(2*x)


x = np.array([0.2387518, 2.8885488, 2.6072087, 3.9613749, 5.7705317,
              3.8547755, 2.1313218, 0.8722102, 3.1486855, 2.8349465])
noise = np.array([0.11764006, -0.05925826, 0.04037102, 0.12340885,
                  0.05293883, 0.17970784,  0.01812706,  0.10901434,
                  -0.0482645, -0.30808059])
y = np.sin(2*x) + noise

x_pred = np.linspace(1, 6, p)
mx = np.zeros_like(x_pred)
y_diff = y - 0
var_par = 0.98997829
rho_par = 1.42263276
noise_par = 0.03835484


def cov_fun(x, y):
    gp_cov = utils.sq_exp_cov(x=x, y=y, var=var_par, rho=rho_par)
    return gp_cov


vnn = cov_fun(x=x, y=x) + np.diag(np.ones_like(x) * noise_par)
vnp = cov_fun(x=x, y=x_pred)
vpp = cov_fun(x=x_pred, y=x_pred)
chol_nn = np.linalg.cholesky(vnn)

gp_params = {
    "mean_fun": lambda u: 0 * np.ones_like(u),
    "mean_params": {},
    "cov_fun": cov_fun,
    "cov_params": {}}
noise_params = {
    "cov_fun": lambda u: np.diag(noise_par * np.ones(len(u))),
    "cov_params": {}}

maxiter = 5
chol_shell, y_diff_shell, x_shell, n_list = gp_utils.gp_setup(
        gp_params, x, y, maxiter=maxiter, noise_params=noise_params)


def test_gp_setup():
    assert chol_shell.shape[1] == maxiter + n
    assert chol_shell.shape[0] == maxiter + n
    assert chol_shell[n, n] == 0
    assert y_diff_shell.shape[0] == maxiter + n
    assert x_shell.shape[0] == maxiter + n
    assert isinstance(n_list, list)
    assert n_list[0] == n


def test_sample_then_update():
    _ = gp_utils._sample_then_update(1, x_shell, y_diff_shell, chol_shell,
                                     gp_params, n_list)
    assert chol_shell.shape[1] == maxiter + n
    assert chol_shell.shape[0] == maxiter + n
    assert chol_shell[n, n] != 0
    assert y_diff_shell.shape[0] == maxiter + n
    assert x_shell.shape[0] == maxiter + n
    assert isinstance(n_list, list)
    assert n_list[0] == n + 1


def test_mvn_cond_dist():
    prior_0 = mx == np.zeros_like(x_pred)
    assert prior_0.all()

    cond_mean, cond_cov = gp_utils.mvn_cond_dist(mx, y_diff,
                                                 vpp, vnp, chol_nn)
    mean_diff = cond_mean - np.array([0.96443368, 0.63158481, -0.79012959])
    assert np.mean(np.abs(mean_diff)) < 1e-8
    cov_diff = cond_cov - cond_cov.T
    assert np.mean(np.abs(cov_diff)) < 1e-8


def test_update_chol():
    vpp_sub = vpp[:-1, :-1]
    chol_sub = np.linalg.cholesky(vpp_sub)
    chol_full = np.linalg.cholesky(vpp)
    chol_updated2full = gp_utils.update_chol(
        chol_sub, np.array(vpp[-1, -1]).reshape(1, 1),
        vpp[-1, :-1].reshape(-1, 1))
    assert np.sum(np.abs(chol_full - chol_updated2full)) < 1e-12
