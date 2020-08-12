import pandas as pd
import numpy as np

import default_gp_utils as utils
from scipy.optimize import minimize
from gp_cond_samp import samp_opt

import matplotlib.pyplot as plt


n = 100
low = 0
cap = 6.48


def y_fun(x):
    return np.sin(2 * x)


x = np.random.uniform(low, cap, n)
y = y_fun(x) + np.random.normal(0, 0.2, len(x))

var_init = np.var(y)
noise_init = np.var(y) / 4
d_mat = utils.dist_matrix(x, x)
rho_init = 3.14 / 2
mu_init = np.mean(y)
mle_params = minimize(utils.neg_log_likelihood_gp,
                      [var_init, rho_init, mu_init, noise_init],
                      args=(d_mat, y, utils.sq_exp_cov),
                      bounds=[(1e-8, None),
                              (1e-8, None),
                              (-2, 2),
                              (1e-8, None)],
                      method='trust-constr',
                      tol=1e-5)
init_x = x[np.where(y == np.min(y))[0]]

# get posterior samples
gp_params = {
    "mean_fun": lambda u: mle_params.x[2] * np.ones_like(u),
    "mean_params": {},
    "cov_fun": utils.sq_exp_cov,
    "cov_params": {"var": mle_params.x[0], "rho": mle_params.x[1]}}
noise_params = {
    "cov_fun": lambda u: np.diag(mle_params.x[3] * np.ones(len(u))),
    "cov_params": {}}

# Visualize the setup
poss_x = np.linspace(low, cap, 1000)
poss_y = y_fun(poss_x)
plt.plot(poss_x, poss_y)
plt.scatter(x, y)
plt.plot([init_x[0][0], init_x[0][0]], [-1, 1], color='red')
plt.show()


a = samp_opt(minimize,
             init_x,
             x_past=x,
             eval_past=y,
             gp_params=gp_params,
             noise_params=noise_params)

x_pred = []
y_pred = []
for i in a['sampling_track']:
    x_pred.append(i['x'])
    y_pred.append(i['y'])

plt.plot(poss_x, poss_y)
plt.scatter(poss_x, post_eval, color='black')
plt.plot(poss_x, cond_mean, color='red')
plt.scatter(x, y)
#plt.scatter(x_pred, y_pred, color='black')
plt.plot([init_x[0][0], init_x[0][0]], [-1, 1], color='red')
plt.show()
plt.close()


# 2 D example - Scallop Catches
# Load data
# This data came from the SemiPar library in R
# Ref: http://matt-wand.utsacademics.info/webspr/scallop.html
# Ruppert, D., Wand, M.P. and Carroll, R.J. (2003)
# Semiparametric Regression Cambridge University Press.
# http://stat.tamu.edu/~carroll/semiregbook/

# TODO: switch to using csv.reader()
scallops = pd.read_csv('scallop.csv')
scallops.rename(columns={'tot.catch': 'tot_catch'}, inplace=True)
scatter = plt.scatter(scallops.longitude, scallops.latitude,
                      c=np.log(scallops['tot_catch']+1), cmap='OrRd')
plt.legend(*scatter.legend_elements(), title='log(total catches + 1)')
plt.show()

# Ignore issues related to projection
x = scallops[['longitude', 'latitude']].values
y = scallops['tot_catch'].values
y.resize(x.shape[0], 1)
# negative since we are minimizing the surface
nlogy = -np.log(y + 1)

# estimate initial parameters
var_init = np.var(nlogy)
d_mat = utils.dist_matrix(x, x)
rho_init = np.percentile(d_mat, 10)
mu_init = np.mean(nlogy)
mle_params = minimize(utils.neg_log_likelihood_exp_gp,
                      [var_init, rho_init, mu_init],
                      args=(d_mat, nlogy),
                      bounds=[(1e-8, None),
                              (np.min(d_mat[d_mat > 0]), np.max(d_mat)),
                              (np.min(nlogy), 0)]
                      )
init_x = x[np.where(y == np.max(y))[0]]

# get posterior samples
gp_params = {
    "mean_fun": lambda u: mle_params.x[2],
    "mean_params": {},
    "cov_fun": utils.exp_cov,
    "cov_params": {"var": mle_params.x[0], "rho": mle_params.x[1]}}
noise_var = np.std(nlogy) / 2
noise_params = {
    "cov_fun": lambda u: np.diag(noise_var * np.ones(len(u))),
    "cov_params": {}}

a = samp_opt(minimize,
             init_x,
             x_past=x,
             eval_past=nlogy,
             gp_params=gp_params,
             noise_params=noise_params)

# plot results
y_pred = []
x_pred = []
for i in a['sampling_track']:
    x_pred.append(i['x'])
    y_pred.append(i['y'])

x_pred = np.concatenate(x_pred, axis=0)
y_pred = np.concatenate(y_pred)

scatter = plt.scatter(scallops.longitude, scallops.latitude,
                      c=np.log(scallops['tot_catch']+1), cmap='OrRd')
scatter = plt.scatter(x_pred[:, 0], x_pred[:, 1],
                      c=y_pred.reshape(-1), cmap='Purples')
plt.legend(*scatter.legend_elements(), title='log(total catches + 1)')
plt.show()


