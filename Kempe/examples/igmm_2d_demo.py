#!/usr/bin/env python

"""
A basic demo of 2D generated data for illustrating the IGMM.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2013, 2014
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

sys.path.append("..")

from bayes_gmm.niw import NIW
from bayes_gmm.igmm import IGMM
from plot_utils import plot_ellipse, plot_mixture_model

logging.basicConfig(level=logging.INFO)

random.seed(1)
np.random.seed(1)


def main():

    # Data parameters
    D = 2           # dimensions
    N = 500         # number of points to generate
    K_true = 5      # the true number of components

    # Model parameters
    alpha = 10
    K = 3           # initial number of components
    n_iter = 20
    r = 1.

    # Generate data
    mu_scale = 4.0
    covar_scale = 0.3
    z_true = np.random.randint(0, K_true, N)
    mu = np.random.randn(D, K_true)*mu_scale
    X = mu[:, z_true] + np.random.randn(D, N)*covar_scale
    X = X.T

    # Intialize prior
    m_0 = np.zeros(D)
    k_0 = covar_scale**2/mu_scale**2
    v_0 = D + 3
    S_0 = covar_scale**2*v_0*np.eye(D)
    prior = NIW(m_0, k_0, v_0, S_0)

    # Setup IGMM
    igmm = IGMM(X, prior, alpha, assignments="rand", K=K, r=r)
    # igmm = IGMM(X, prior, alpha, assignments="one-by-one", K=K)

    # Perform Gibbs sampling
    record = igmm.gibbs_sample(n_iter)

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_mixture_model(ax, igmm)
    for k in range(igmm.components.K):
        mu, sigma = igmm.components.rand_k(k)
        plot_ellipse(ax, mu, sigma)

    plt.show()


if __name__ == "__main__":
    main()
