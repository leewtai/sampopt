import numpy as np
from scipy.linalg import cho_factor, solve_triangular


def mvn_cond_dist(mx, y_dev, vxx, vyx, vyy):
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
    chol_vyy, lower = cho_factor(vyy)
    L_inv_vyx = solve_triangular(chol_vyy, vyx, lower=lower)
    cond_var = vxx - np.matmul(L_inv_vyx.T, L_inv_vyx)

    L_inv_y_dev = solve_triangular(chol_vyy, y_dev, lower=lower)
    cond_mean = mx + np.matmul(L_inv_vyx.T, L_inv_y_dev)

    return cond_mean, cond_var
