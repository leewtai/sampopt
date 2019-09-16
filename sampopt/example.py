import pandas as pd
import numpy as np

import default_gp_utils as utils
from scipy.optimize import minimize
from gp_cond_samp import samp_opt

# Load data
scallops = pd.read_csv('scallop.csv', sep=' ')

x = scallops[['latitude', 'longitude']].values
y = scallops['tot.catch'].values
y.resize((x.shape[0], 1))

# estimate initial parameters
var_init = min(np.mean(y) ** 2, np.var(y))
d_mat = utils.dist_matrix(x, x)
rho_init = np.percentile(d_mat, 10)
mu_init = np.mean(np.log(y + 1))
mle_params = minimize(utils.neg_log_likelihood_exp_gp,
                      [var_init, rho_init, mu_init],
                      args=(d_mat, np.log(y + 1)),
                      bounds=[(1e-8, None),
                              (np.min(d_mat[d_mat > 0]), None),
                              (None, None)]
                      )
init_x = x[np.where(y == np.max(y))[0]]

# get posterior samples
gp_params = {
    "mean_fun": lambda u: mle_params.x[2],
    "mean_params": {},
    "cov_fun": utils.exp_cov,
    "cov_params": {"var": mle_params.x[0], "rho": mle_params.x[1]}}

samp_opt(minimize,
         init_x,
         x_past=x,
         eval_past=np.log(y + 1),
         gp_params=gp_params)

# plot results
