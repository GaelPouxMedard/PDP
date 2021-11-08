from igmm import IGMM, init_prior
import numpy as np
import matplotlib.pyplot as plt
import logging
from Kempe.bayes_gmm.plot_utils import plot_ellipse, plot_mixture_model

# Get data
indic = "Antoni-150-200"
folder = "Data/Antoni/XP/"
with open("Data/Antoni/Antoni-150-200.csv", "r") as f:
    coords = []
    for i, line in enumerate(f):
        if i==0:
            continue
        dat = line.split("\t")
        x,y = float(dat[5]), float(dat[4])
        coords.append([x,y])
    print(coords)
print(len(coords))
coords = np.array(coords)
#plt.scatter(coords[:,0], coords[:,1], s=0.1)
#plt.show()
X = coords


covar_scale = 500000/4
shiftMeans = 500000/2
n_iter = 1000
nbRunsPerR = 10
alpha = 1.

printLogs = False
plotRes = False
allr = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1]
D = 2
K = 1
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
prior = init_prior(shiftMeans, covar_scale, D)
for r in allr:
    tabK = []
    tabYInf = []
    for i in range(nbRunsPerR):
        # Setup IGMM
        igmm = IGMM(X, prior, alpha, assignments="rand", K=K, r=r, printLogs=printLogs)

        # Perform Gibbs sampling
        if printLogs:
            logger.info("Initial log marginal prob: " + str(igmm.log_marg()))
            logger.info("Assignments: " + str(igmm.components.assignments))
        record = igmm.gibbs_sample(n_iter)

        tabK=np.array([igmm.components.K])
        tabYInf=np.array(igmm.components.assignments)
        r = np.round(r, 2)
        print(fr"======= r={r} - Shift={shiftMeans} - Alpha={alpha} - K={np.mean(tabK)}±{np.std(tabK)}")


        np.save(folder + f"{indic} - r={r} - Shift={shiftMeans} - Alpha={alpha} - Run={i} - K", tabK)
        np.save(folder + f"{indic} - r={r} - Shift={shiftMeans} - Alpha={alpha} - Run={i} - Data", X)
        np.save(folder + f"{indic} - r={r} - Shift={shiftMeans} - Alpha={alpha} - Run={i} - TInf", tabYInf)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_mixture_model(ax, igmm)
        plt.title(f"r={r} - Shift={shiftMeans} - Run={i}")
        plt.savefig(folder + f"{indic} - r={r} - Shift={shiftMeans} - Alpha={alpha} - Run={i}_withoutEll.pdf")
        for k in range(igmm.components.K):
            mu, sigma = igmm.components.rand_k(k)
            plot_ellipse(ax, mu, sigma)
        plt.savefig(folder + f"{indic} - r={r} - Shift={shiftMeans} - Alpha={alpha} - Run={i}_withEll.pdf")


        if plotRes:
            plt.show()

        plt.close()
            

print("DONE")