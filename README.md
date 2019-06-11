# sampopt
Obtain posterior samples for the extrema

#### Introduction
Bayesian optimization tackles the problem of obtaining the optimal parameters
when we have noisy samples of an objective function and the cost to evaluate
the objective function is high.

The typical approach is to estimate the objective function via a flexible
statistical model, then estimate the optimal parameters from this model.

However, a common confusion is the difference between

argmin_{x} E[f(x)]

vs

E[argmin_{x} f(x)]

In the former case, people collect samples of the noisy curve $\hat{f}(x)$, 
take the expectation over these samples, then optimize on the deterministic
surface using off-the-shelf optimization routines.

In the latter case, optimization is performed on each noisy curve
separately, then an expectation is taken. Note that this expectation step 
is not advised in multimodal cases but used here to highlight the
difference between the 2 approaches.

Challenges the former method faces is the optimization routine often has to
occur over a pre-determined set of parameter values before we know the
extrema. This is often done by calculating the curves on a fine grid which
can be computationally expensive.

This package is designed to help obtain the samples of the extrema as done
in the latter case using Gaussian Processes and off-the-shelf optimization
routines.


#### Example
###### Example with known objective function

###### Example with historical objective function evaluations

###### Example with noisy historical objective function evaluations
