"""
Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

from numpy.linalg import cholesky, det, inv, slogdet
from scipy.special import logsumexp
from scipy.special import gammaln
import logging
import math
import numpy as np
import time
from sklearn.metrics import normalized_mutual_info_score

try:
    from bayes_gmm.gaussian_components import GaussianComponents
    from bayes_gmm.gaussian_components_diag import GaussianComponentsDiag
    from bayes_gmm.gaussian_components_fixedvar import GaussianComponentsFixedVar
    import bayes_gmm.utils as utils
except:
    from gaussian_components import GaussianComponents
    from gaussian_components_diag import GaussianComponentsDiag
    from gaussian_components_fixedvar import GaussianComponentsFixedVar
    import utils as utils

import matplotlib.pyplot as plt
from plot_utils import plot_ellipse, plot_mixture_model

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                                  IGMM CLASS                                 #
#-----------------------------------------------------------------------------#

class IGMM(object):
    """
    An infinite Gaussian mixture model (IGMM).

    See `GaussianComponents` for an overview of the parameters not mentioned
    below.

    Parameters
    ----------
    alpha : float
        Concentration parameter for the Dirichlet process.
    assignments : vector of int or str
        If vector of int, this gives the initial component assignments. The
        vector should therefore have N entries between 0 and `K`. Values of
        -1 is also allowed, indicating that the data vector does not belong to
        any component. Alternatively, `assignments` can take one of the
        following values:
        - "rand": Vectors are assigned randomly to one of `K` components.
        - "one-by-one": Vectors are assigned one at a time; the value of
          `K` becomes irrelevant.
        - "each-in-own": Each vector is assigned to a component of its own.
    K : int
        The initial number of mixture components: this is only used when
        `assignments` is "rand".
    covariance_type : str
        String describing the type of covariance parameters to use. Must be
        one of "full", "diag" or "fixed".
    """

    def __init__(self, X, prior, alpha, assignments="rand", K=1, K_max=None, r=1., covariance_type="full", printLogs=True):
        self.alpha = alpha
        self.r = r
        self.printLogs = printLogs
        N, D = X.shape

        # Initial component assignments
        if assignments == "rand":
            assignments = np.random.randint(0, K, N)

            # Make sure we have consequetive values
            for k in range(assignments.max()):
                while len(np.nonzero(assignments == k)[0]) == 0:
                    assignments[np.where(assignments > k)] -= 1
                if assignments.max() == k:
                    break
        elif assignments == "one-by-one":
            assignments = -1*np.ones(N, dtype="int")
            assignments[0] = 0  # first data vector belongs to first component
        elif assignments == "each-in-own":
            assignments = np.arange(N)
            pass
        else:
            # assignments is a vector
            pass

        if covariance_type == "full":
            self.components = GaussianComponents(X, prior, assignments, K_max)
        elif covariance_type == "diag":
            self.components = GaussianComponentsDiag(X, prior, assignments, K_max)
        elif covariance_type == "fixed":
            self.components = GaussianComponentsFixedVar(X, prior, assignments, K_max)
        else:
            assert False, "Invalid covariance type."

    def log_marg(self):
        """Return log marginal of data and component assignments: p(X, z)"""

        # Log probability of component assignment P(z|alpha)
        # Equation (10) in Wood and Black, 2008
        # Use \Gamma(n) = (n - 1)!
        facts_ = gammaln(self.components.counts[:self.components.K])
        facts_[self.components.counts[:self.components.K] == 0] = 0  # definition of log(0!)
        log_prob_z = (
            (self.components.K - 1)*math.log(self.alpha) + gammaln(self.alpha)
            - gammaln(np.sum(self.components.counts[:self.components.K])
            + self.alpha) + np.sum(facts_)
            )

        log_prob_X_given_z = self.components.log_marg()

        return log_prob_z + log_prob_X_given_z

    # @profile
    def gibbs_sample(self, n_iter):
        """
        Perform `n_iter` iterations Gibbs sampling on the IGMM.

        A record dict is constructed over the iterations, which contains
        several fields describing the sampling process. Each field is described
        by its key and statistics are given in a list which covers the Gibbs
        sampling iterations. This dict is returned.
        """

        # Setup record dictionary
        record_dict = {}
        record_dict["sample_time"] = []
        start_time = time.time()
        record_dict["log_marg"] = []
        record_dict["components"] = []

        # Loop over iterations
        for i_iter in range(n_iter):

            # Loop over data items
            import random
            permuted = list(range(self.components.N))
            random.shuffle(permuted)
            for i in permuted:
            #for i in range(self.components.N):

                # Cache some old values for possible future use
                k_old = self.components.assignments[i]
                K_old = self.components.K
                stats_old = self.components.cache_component_stats(k_old)

                # Remove data vector `X[i]` from its current component
                self.components.del_item(i)

                # Compute log probability of `X[i]` belonging to each component
                log_prob_z = np.zeros(self.components.K + 1, np.float)
                # (25.35) in Murphy, p. 886
                log_prob_z[:self.components.K] = np.log(self.components.counts[:self.components.K] ** self.r)
                # (25.33) in Murphy, p. 886
                log_prob_z[:self.components.K] += self.components.log_post_pred(i)
                # Add one component to which nothing has been assigned
                log_prob_z[-1] = math.log(self.alpha) + self.components.cached_log_prior[i]
                prob_z = np.exp(log_prob_z - logsumexp(log_prob_z))

                # Sample the new component assignment for `X[i]`
                k = utils.draw(prob_z)
                # logger.debug("Sampled k = " + str(k) + " from " + str(prob_z) + ".")

                # Add data item X[i] into its component `k`
                if k == k_old and self.components.K == K_old:
                    # Assignment same and no components have been removed
                    self.components.restore_component_from_stats(k_old, *stats_old)
                    self.components.assignments[i] = k_old
                else:
                    # Add data item X[i] into its new component `k`
                    self.components.add_item(i, k)

            # Update record
            record_dict["sample_time"].append(time.time() - start_time)
            start_time = time.time()
            record_dict["log_marg"].append(self.log_marg())
            record_dict["components"].append(self.components.K - 1)

            # Log info
            info = "iteration: " + str(i_iter)
            for key in sorted(record_dict):
                info += ", " + key + ": " + str(record_dict[key][-1])
            info += "."
            if self.printLogs:
                logger.info(info)

        return record_dict


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#
def oldGen():
    unbalanced = False
    mu_scale = 4.0
    covar_scale = 0.7
    if unbalanced:
        arrProbs = np.random.random((K_true))
        arrProbs /= np.sum(arrProbs)
        z_true = np.random.choice(list(range(K_true)), size=N, p=arrProbs)
    else:
        z_true = np.random.randint(0, K_true, N)
    mu = np.random.randn(D, K_true) * mu_scale
    X = mu[:, z_true] + np.random.randn(D, N) * covar_scale
    X = X.T

    return X

def init_prior(mu_scale, covar_scale, D):
    m_0 = np.zeros(D)
    k_0 = covar_scale ** 2 / mu_scale ** 2
    v_0 = D + 3
    S_0 = covar_scale ** 2 * v_0 * np.eye(D)
    prior = NIW(m_0, k_0, v_0, S_0)
    return prior

def generateData(N, K_true, mu, sig):
    unbalanced = False

    if unbalanced:
        arrProbs = np.random.random((K_true))
        arrProbs /= np.sum(arrProbs)
        z_true = np.random.choice(list(range(K_true)), size=N, p=arrProbs)
    else:
        z_true = np.random.randint(0, K_true, N)

    X = mu[z_true, :] + sig[z_true, :] * np.random.randn(N, 2)

    return X, z_true


import random

from niw import NIW

logging.basicConfig(level=logging.INFO)

random.seed(1)
np.random.seed(1)

# Data parameters
D = 2           # dimensions
N = 1000        # number of points to generate
K_true = 5      # the true number of components
mu_scale, covar_scale = 1., 1.
means = np.array([[-2,0], [-1,0], [0,0], [1,0], [2,0]])
sigmas = np.ones((K_true, 2))**0

X, z_true = generateData(N, K_true, means, sigmas)

print(np.unique(z_true, return_counts=True))

# Model parameters
alpha = 1.
K = 1  # initial number of components
n_iter = 200
nbRunsPerR = 1

printLogs=False
plotRes=False


folder = "Data/XP/"
for shiftMeans in np.linspace(0.+1e-20, 10, 11):
    meansShifted = means * shiftMeans
    prior = init_prior(shiftMeans, covar_scale, D)
    X, z_true = generateData(N, K_true, meansShifted, sigmas)
    if plotRes and True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        igmm = IGMM(X, prior, alpha, assignments="rand", K=K, r=1., printLogs=printLogs)
        plot_mixture_model(ax, igmm)
        plt.show()

    for r in np.linspace(0.8, 1.2, 11):
        tabNMI, tabK = [], []
        tabYTrue, tabYInf = [], []
        for i in range(nbRunsPerR):
            #print(r, i)
            # Setup IGMM
            igmm = IGMM(X, prior, alpha, assignments="rand", K=K, r=r, printLogs=printLogs)

            # Perform Gibbs sampling
            if printLogs:
                logger.info("Initial log marginal prob: " + str(igmm.log_marg()))
                logger.info("Assignments: " + str(igmm.components.assignments))
                logger.info("True-------: " + str(z_true))
            record = igmm.gibbs_sample(n_iter)

            if plotRes:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plot_mixture_model(ax, igmm)
                for k in range(igmm.components.K):
                    mu, sigma = igmm.components.rand_k(k)
                    plot_ellipse(ax, mu, sigma)
                plt.title(f"r={r} - Shift={shiftMeans}")
                plt.show()


            NMI = normalized_mutual_info_score(z_true, igmm.components.assignments)
            K = igmm.components.K
            tabNMI.append(NMI)
            tabK.append(K)

            tabYTrue.append(z_true)
            tabYInf.append(igmm.components.assignments)


        print(fr"======= r={r} - NMI={np.mean(tabNMI)}±{np.std(tabNMI)} - K={np.mean(tabK)}±{np.std(tabK)}")

        tabYTrue, tabYInf = np.array(tabYTrue), np.array(tabYInf)

        np.save(folder + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - NMI", tabNMI)
        np.save(folder + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - K", tabK)
        np.save(folder + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - Data", X)  # CHANGER
        np.save(folder + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - YTrue", tabYTrue)
        np.save(folder + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - TInf", tabYInf)

        print(fr"======= r={r} - Shift={shiftMeans} - Alpha={alpha} - NMI={np.mean(resTmp)}±{np.std(resTmp)} - K={np.mean(Ktmp)}±{np.std(Ktmp)}")
