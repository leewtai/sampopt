import numpy as np
from scipy.linalg import cho_factor, solve_triangular


def mvn_cond_dist(mx, y_dev, vxx, vyx, chol_vyy):
    """ Get a sample from the conditional distribution of P(x|y)
    where (x, y) is jointly as a MultiVarNormal(m, V).
    We assume V is a block matrix of [vxx, vxy,
                                      vyx, vyy]
    and the mean m = (mx, my).

    Args:
        mx (narray): the prior mean array for x.
        y_dev (narray): the array (y - my).
        vxx (narray): upper left block matrix of the joint covariance matrix
        vyx (narray): upper right block matrix of the joint covariance matrix
        vyy (narray): lower right block matrix of the joint covariance matrix

    Returns:
        cond_mean (narray): conditional mean vector for x
        cond_var (narray): conditional variance matrix for x
    """
    # get the lower triangular matrix, L, of vyy = L*conj_trans(L) from the
    # cholesky decomposition
    chol_inv_vyx = solve_triangular(chol_vyy, vyx, lower=True)
    cond_var = vxx - np.matmul(chol_inv_vyx.T, chol_inv_vyx)

    chol_inv_y_dev = solve_triangular(chol_vyy, y_dev, lower=True)
    cond_mean = mx + np.matmul(chol_inv_vyx.T, chol_inv_y_dev)

    return cond_mean, cond_var, chol_vyy


def update_chol(chol_old, cov_new, cross_cov):
    """ Updates the cholesky matrix with the addition of a new data point.
    If L_o is the original cholesky matrix, and denoting the covariance
    matrix with the new data points as A. Then adding the new data point
    would result in the
    L_f = [L_o          0
           C*L_o^(-1)   A^(1/2)]
    """
    chol_new, lower_new = cho_factor(cov_new, lower=True)
    #TODO, make sure cross_cov is correct
    chol_cross = solve_triangular(chol_new.T, cross_cov, lower=False)

    chol_left = np.concatenate(chol_old, chol_cross.T, axis=0)
    chol_right = np.concatenate((np.zero(chol_old.shape[0]),
                                 chol_new),
                                axis=0)
    chol_full = np.concatenate(chol_left, chol_right, axis=1)

    return chol_full


def samp_opt(minimizer, x0, x_past=[], eval_past=[], gp_params={}):
    init_cov = gp_params['cov_fun'](x_past, x_past, **gp_params['cov_params'])
    # For known objective functions, you would define this as the mean_fun
    y_dev = eval_past - gp_params['mean_fun'](x_past,
                                              **gp_params['mean_params'])
    chol, lower = cho_factor(init_cov, lower=True)

    def fun(x):
        mx = gp_params['mean_fun'](x, **gp_params['mean_params'])
        vxx = gp_params['cov_fun'](x, x, **gp_params['cov_params'])
        vyx = gp_params['cov_fun'](x_past, x, **gp_params['cov_params'])

        cond_mean, cond_cov, chol = mvn_cond_dist(
            mx, y_dev, vxx, vyx, chol)
        post_eval = np.random.multivariate_normal(cond_mean, cond_cov)
        x_past.append(x)
        y_dev.append(post_eval - mx)
        chol = update_chol(chol, vxx, vyx)

        return post_eval

    minimizer(fun, x0, args=fun_pos_args)
