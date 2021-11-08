from igmm import IGMM, init_prior
import numpy as np
import matplotlib.pyplot as plt
import logging
from Kempe.bayes_gmm.plot_utils import plot_ellipse, plot_mixture_model
import sys
from scipy.spatial import distance_matrix
import os
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, v_measure_score, fowlkes_mallows_score
from scipy.stats import sem


def variation_of_information(X, Y):
  n = float(sum([len(x) for x in X]))
  sigma = 0.
  for x in X:
    p = len(x) / n
    for y in Y:
      q = len(y) / n
      r = len(set(x) & set(y)) / n

      if r > 0.0:
        sigma += - r * (np.log(r / p) + np.log(r / q))
  return abs(sigma)



from sklearn import datasets

listDS = [
          ("breast_cancer", datasets.load_breast_cancer()),
          ("digits", datasets.load_digits()),
          ("iris", datasets.load_iris()),
          #("boston", datasets.load_boston()),
          #("diabetes", datasets.load_diabetes()),
          ("wine", datasets.load_wine()),
          ]

for nameDS, DS in listDS:
    X = DS.data
    folder = "Data/RW/XP/"+nameDS
    if not os.path.exists(folder):
        os.makedirs(folder)
    print(DS.target)

    #dists = distance_matrix(X, X)
    #shiftMeans = np.mean(dists[dists!=0])
    #covar_scale = np.mean(dists[dists!=0])/100
    shiftMeans = 1
    covar_scale = 0.1
    print(shiftMeans)
    print(covar_scale)
    print(np.mean(X, axis=0))
    print(np.std(X, axis=0))

    n_iter = 5000
    if nameDS=="digits": n_iter = 1000
    nbRunsPerR = 100
    alpha = 1.
    D = len(X[-1])


    printLogs = True
    plotRes = False
    allr = [1., 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1]
    K = 10
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    prior = init_prior(shiftMeans, covar_scale, D)
    with open(folder + f"/_Results.txt", "w+") as f:
        f.write("r\trun\tAdjMI\tAdjRand\tNVI\tVmeas\tFowlkes\tNMI\tvarK\n")
    with open(folder + f"/_Results_mean.txt", "w+") as f:
        f.write("r\tAdjMI\tAdjRand\tNVI\tVmeas\tFowlkes\tNMI\tvarK\n")
    with open(folder + f"/_Results_std.txt", "w+") as f:
        f.write("r\tAdjMI\tAdjRand\tNVI\tVmeas\tFowlkes\tNMI\tvarK\n")
    with open(folder + f"/_Results_sem.txt", "w+") as f:
        f.write("r\tAdjMI\tAdjRand\tNVI\tVmeas\tFowlkes\tNMI\tvarK\n")

    for r in allr:
        tabK = []
        tabYInf = []
        tabRes = []
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


            tabYTrue = DS.target

            partTrue, partInf = [], []
            for c in set(tabYTrue):
                partTrue.append(list(np.where(tabYTrue==c)[0]))
            for c in set(tabYInf):
                partInf.append(list(np.where(tabYInf==c))[0])

            maxVI = np.log(len(tabYTrue))
            AdjRand = adjusted_rand_score(tabYTrue, tabYInf)
            AdjMI = adjusted_mutual_info_score(tabYTrue, tabYInf)
            NVI = variation_of_information(partTrue,partInf) / maxVI
            NMI = normalized_mutual_info_score(tabYTrue, tabYInf)
            Vmeas = v_measure_score(tabYTrue, tabYInf)
            Fowlkes = fowlkes_mallows_score(tabYTrue, tabYInf)
            varK = np.abs((len(set(tabYInf))-len(set(tabYTrue)))/len(set(tabYTrue)))

            print(f"{AdjMI}\t{AdjRand}\t{NVI}\t{Vmeas}\t{Fowlkes}\t{NMI}\t{varK}\n")
            tabRes.append((AdjMI, AdjRand, NVI, Vmeas, Fowlkes, NMI, varK))
            with open(folder + f"/_Results.txt", "a+") as f:
                #f.write(f"{r}\t{i}\t{AdjMI}\t{AdjRand}\t{NVI}\t{Vmeas}\t{Fowlkes}\t{NMI}\t{varK}\n")
                txt = "\t".join(map(str, tabRes[-1]))
                f.write(f"{r}\t{i}\t{txt}\n")


            np.save(folder + f"/r={r} - Shift={shiftMeans} - Alpha={alpha} - Run={i} - K", tabK)
            np.save(folder + f"/r={r} - Shift={shiftMeans} - Alpha={alpha} - Run={i} - Data", X)
            np.save(folder + f"/r={r} - Shift={shiftMeans} - Alpha={alpha} - Run={i} - TInf", tabYInf)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot_mixture_model(ax, igmm)
            plt.title(f"r={r} - Shift={shiftMeans} - Run={i}")
            plt.savefig(folder + f"/r={r} - Shift={shiftMeans} - Alpha={alpha} - Run={i}_withoutEll.pdf")
            plt.savefig(folder + f"/r={r} - Shift={shiftMeans} - Alpha={alpha} - Run={i}_withEll.pdf")


            if plotRes:
                plt.show()

            plt.close()


        tabRes = np.array(tabRes)
        "AdjMI\tAdjRand\tNVI\tVmeas\tFowlkes\tNMI\tvarK\n"
        mean, std, sem_val = np.mean(tabRes, axis=0), np.std(tabRes, axis=0), sem(tabRes, axis=0)
        with open(folder + f"/_Results_mean.txt", "a+") as f:
            #f.write(f"{r}\t{mean[0]}\t{mean[1]}\t{mean[2]}\t{mean[3]}\t{mean[4]}\n")
            txt = "\t".join(map(str, mean))
            f.write(f"{r}\t{txt}\n")
        with open(folder + f"/_Results_std.txt", "a+") as f:
            #f.write(f"{r}\t{std[0]}\t{std[1]}\t{std[2]}\t{std[3]}\t{std[4]}\n")
            txt = "\t".join(map(str, std))
            f.write(f"{r}\t{txt}\n")
        with open(folder + f"/_Results_sem.txt", "a+") as f:
            #f.write(f"{r}\t{sem_val[0]}\t{sem_val[1]}\t{sem_val[2]}\t{sem_val[3]}\t{sem_val[4]}\n")
            txt = "\t".join(map(str, sem_val))
            f.write(f"{r}\t{txt}\n")




    print("DONE")