import numpy as np
from scipy.linalg import cho_factor, solve_triangular


def mvn_cond_dist(mx, y_diff, vxx, vyx, chol_vyy):
    """ Returns a sample from the conditional distribution of P(x|y)
    where (x, y) is jointly as a MultiVarNormal(m, V).
    We assume V is a block matrix of [vxx, vxy,
                                      vyx, vyy]
    and the mean m = (mx, my)^T.

    Args:
        mx (narray): the prior mean array for x.
        y_diff (narray): the array (y - my).
        vxx (narray): upper left block matrix of the joint covariance matrix
        vyx (narray): lower left block matrix of the joint covariance matrix
        vyy (narray): lower right block matrix of the joint covariance matrix

    Returns:
        cond_mean (narray): conditional mean vector for x
        cond_var (narray): conditional variance matrix for x

    Details:
        The conditional mean follows mx + vxy vyy^{-1} (y - my)
        If chol_vyy.dot(chol_vyy.T) = vyy where chol_vyy is lower triangular,
        vyy^{-1} = np.linalg.inv(chol_vyy.T).dot(np.linalg.inv(chol_vyy))
        then we want
        mx + np.matmul(vxy.dot(np.linalg.inv(chol_vyy.T)),
                       np.linalg.inv(chol_vyy), y_diff)
        Notice that
        vxy.dot(np.linalg.inv(chol_vyy.T))
        = np.linalg.inv(chol_vyy).dot(np.vyx).T
    """
    # get the lower triangular matrix, L, of vyy = L*conj_trans(L) from the
    # cholesky decomposition
    chol_inv_vyx = solve_triangular(chol_vyy, vyx, lower=True)
    cond_var = vxx - np.matmul(chol_inv_vyx.T, chol_inv_vyx)

    chol_inv_y_diff = solve_triangular(chol_vyy, y_diff, lower=True)
    cond_mean = mx + np.matmul(chol_inv_vyx.T, chol_inv_y_diff)

    return cond_mean, cond_var


def update_chol(chol_old, cov_new, cross_cov):
    """ Updates the cholesky matrix with the addition of a new data point.
    If L_o is the original cholesky matrix, and denoting the covariance
    matrix with the new data points as A. Then adding the new data point
    would result in the
    L_f = [L_o          0
           C*L_o^(-1)^T   A^(1/2)]
    L_f * L_f^T = [L_o*L_o^T     C^T
                   C             A + C*L_o^(-1)^T*L_o^(-1)*C^T]
    """
    # First form [L_o   0]
    chol_f_top = np.concatenate((chol_old, np.zeros((chol_old.shape[0], 1))),
                                axis=1)

    # Form the new terms [C*L_o^(-1)^T   A^(1/2)]
    chol_cross = solve_triangular(chol_old, cross_cov, lower=True)
    try:
        chol_new, _ = cho_factor(cov_new - chol_cross.T @ chol_cross,
                                 lower=True)
    except np.linalg.LinAlgError:
        return chol_old

    chol_f_bot = np.concatenate((chol_cross.T, chol_new), axis=1)

    chol_f = np.concatenate((chol_f_top,
                             chol_f_bot),
                            axis=0)

    return chol_f


def _sample_then_update(x_new, x_shell, y_diff_shell, chol_shell,
                        gp_params, n_list, sampling_track=[]):
    """ Returns a sample from the predictive distribution at x given the
    data from x_old then updates the objective function, inplace,
    so the predictive distribution is from [x_old, x]^T.

    For details, see the derivation for equation 2.23, 2.24
    on http://www.gaussianprocess.org/gpml/chapters/RW2.pdf

    Args:
        x (narray): This should be provided by the optimization algorithm
        x_old (narray): This should be the location which the predictive
            distribution is based on.
        y_diff (narray): This should be the difference between the expectation
            vs the realized values in our data.
        chol (np.linalg.chol): the lower triagular matrix by performing
            cholesky decomposition on x_old locations using the cov_fun
            from gp_params.
        gp_params (dict): see details in samp_opt()
        n_list (list):
    Returns:
        y: A random sample from the predictive distribution, this is NOT
            the expectation of the predictive distribution.
    """
    n = n_list[0]
    x_old = x_shell[:n, :]
    y_diff = y_diff_shell[:n, :]
    chol = chol_shell[:n, :n]

    x_new = np.array(x_new).reshape(1, -1)

    mx = gp_params['mean_fun'](x_new, **gp_params['mean_params'])
    # This intentionally does not include the noise
    vxx = gp_params['cov_fun'](x=x_new, y=x_new, **gp_params['cov_params'])
    vyx = gp_params['cov_fun'](x=x_old, y=x_new, **gp_params['cov_params'])

    # This can be sped up
    cond_mean, cond_cov = mvn_cond_dist(mx, y_diff, vxx, vyx, chol)
    post_eval = np.random.multivariate_normal(cond_mean.flatten(), cond_cov)

    sampling_track.append({'x': x_new, 'y': post_eval, 'cond_mean': cond_mean})

    # numerical issues will first happen with the cholesky update
    new_chol = update_chol(chol, vxx, vyx)
    if new_chol.shape == chol.shape:
        return cond_mean

    n1 = n + 1
    chol_shell[:n1, :n1] = new_chol
    x_shell[:n1, :] = np.concatenate((x_old, x_new), 0)
    y_diff_n1 = post_eval.reshape(cond_mean.shape) - mx
    y_diff_shell[:n1, :] = np.concatenate((y_diff, y_diff_n1))
    # This is necessary so the n_list object changes the behavior of other
    # iterations in the sampling, overwriting n_list entirely would not have
    # the desired effect
    n_list[0] = n1

    return post_eval[0]


def samp_opt(minimizer, x0, x=[], y=[], gp_params={},
             maxiter=500,
             noise_params={'cov_fun': lambda x: np.diag(0 * np.ones(len(x)))},
             opt_args={}):
    """ Sets up and runs the Bayesian optimization using the provided
    optimizer.

    Args:
        minimizer: the optimizer like from scipy.optimize.minimize
        x0 (narray): starting value for the optimizer
        x_past (narray): historical locations (parameter combinations)
            that have been evaluated
        eval_past (narray): at the corresponding locations in x_past,
            the resulting evaluation of the objective function. This
            can be noisy, i.e. evaluating the same location may not
            provide the same outcome.
        gp_params (dict): dictionary that contains the covariance function
            (key is 'cov_fun'), its parameters (key as 'cov_params'),
            the mean function (key as 'mean_fun'), and its parameters (
            key as 'mean_params'). The first argument for these functions
            must be x_past or larger.
        maxiter (int): maximum number of evaluations. This is used for
            pre-allocating a large enough numpy array to speed up
            calculations.
        noise_params (dict): dictionary that contains the covariance
            function (key is 'cov_fun'), its parameters (key as 'cov_params').
            The first argument for these functions must be x_past or larger.
    Returns:
        cond_mean (narray): conditional mean vector for x
        cond_var (narray): conditional variance matrix for x
    """
    # For non-stochastic objective functions, define that as the mean_fun
    chol_shell, y_diff_shell, x_shell, n_list = gp_setup(gp_params, x,
                                                         y, maxiter,
                                                         noise_params)
    sampling_track = []

    opt_out = minimizer(_sample_then_update,
                        x0,
                        args=(x_shell, y_diff_shell, chol_shell, gp_params,
                              n_list, sampling_track),
                        **opt_args)
    return {'opt_out': opt_out, 'sampling_track': sampling_track}


def gp_setup(gp_params, x, y, maxiter, noise_params):

    # We can allocate a larger than necessary matrix and vector
    # so later iterations do not overwrite the underlying components
    n = len(y)
    chol_shell = np.zeros((n + maxiter, n + maxiter))
    y_diff_shell = np.empty((n + maxiter, 1))
    if len(x.shape) < 2:
        x = x.reshape(-1, 1)
    x_shell = np.empty((n + maxiter, x.shape[1]))

    cov = gp_params['cov_fun'](x=x, y=x, **gp_params['cov_params'])
    noise_cov = noise_params['cov_fun'](x, **noise_params['cov_params'])
    # This is the only time we'll add the noise values
    chol, _ = cho_factor(cov + noise_cov, lower=True)
    chol_shell[:n, :n] = chol

    y_expect = gp_params['mean_fun'](x, **gp_params['mean_params'])
    y_diff = y.reshape((-1, 1)) - y_expect
    y_diff_shell[:n, :] = y_diff

    x_shell[:n, :] = x

    return chol_shell, y_diff_shell, x_shell, [n]
