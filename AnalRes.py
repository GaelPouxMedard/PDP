import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from scipy.stats import sem
import seaborn as sns

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'


def plotKvN(folder, a):
    gdN = int(1e7)
    tabKTot = {}
    tabNTot = {}
    for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2]:
        print(r)
        nbRuns = 100
        tabN = np.load(f"Data/DistribK/N_{r}_{a}_{gdN}_{0}.npy")
        tabNTot[r] = tabN[1:]
        tabKTot[r] = []
        for run in range(nbRuns):
            tabK = np.load(f"Data/DistribK/K_{r}_{a}_{gdN}_{run}.npy")
            with open(f"Data/DistribK/pop_{r}_{a}_{gdN}_{run}.txt", "r") as f:
                dicPopN = eval(f.read())
            popN = np.array(dicPopN[tabN[-1]][1:])

            tabK = np.array(tabK)
            tabKTot[r].append(tabK)

    colors = iter(reversed([plt.cm.tab20(i) for i in range(20)]*2))
    for i, r in enumerate(tabKTot):
        tabKTot[r] = np.array(tabKTot[r])[:, 1:]
        meanTrue, stdTrue = np.mean(tabKTot[r], axis=0), np.std(tabKTot[r], axis=0)
        if r!=1:
            #tabPred = a*(tabNTot[r] ** ((1 - r)) - 1) / (1 - r)
            if r<1:
                tabPred = 2*a**(1/2-(r**2)/2 + r)*(tabNTot[r] ** ((1 - r**2)/2) - 1) / (1 - r**2)
            else:
                tabPred = a * (tabNTot[r] ** (1 - r) - 1) / (1 - r)
        else:
            tabPred = a*np.log(tabNTot[r])
        shift = 10**(i+1)

        norm = 1
        shift = 1./10**(i+1)
        stdTrue = (stdTrue/np.max(meanTrue)**norm)*shift
        meanTrue = (meanTrue/np.max(meanTrue)**norm)*shift
        tabPred = (tabPred/np.max(tabPred)**norm)*shift
        ctmp = next(colors)
        plt.plot(tabNTot[r], tabPred, color=next(colors), label=f"r={r}")
        plt.plot(tabNTot[r], meanTrue, color=ctmp)
        plt.fill_between(tabNTot[r], meanTrue - stdTrue, meanTrue + stdTrue, color=ctmp, alpha=0.3)

    plt.legend()
    plt.semilogy()
    plt.gca().set_yticklabels([])
    plt.xlabel("N")
    plt.ylabel("K")
    plt.savefig(folder+"Th3.pdf")
    plt.semilogx()
    plt.savefig(folder+"Th3_semilogx.pdf")
    #plt.show()
    plt.close()

def plotEntvr(folder, a):
    gdN = int(1e7)
    tabEnt = {}
    tabNTot = {}
    for r in [0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2]:
        print(r)
        nbRuns = 1
        tabN = np.load(f"Data/DistribK/N_{r}_{a}_{gdN}_{0}.npy")
        tabNTot[r] = tabN
        tabEnt[r] = []
        for run in range(nbRuns):
            with open(f"Data/DistribK/pop_{r}_{a}_{gdN}_{run}.txt", "r") as f:
                dicPopN = eval(f.read())
            tmp = []

            for n in dicPopN:
                popN = np.array(dicPopN[n][1:])**r
                popN /= np.sum(popN)
                e = -np.sum((popN)*np.log(popN))
                e /= np.log(len(popN))
                tmp.append(e)

            tabEnt[r].append(tmp)

    colors = iter([plt.cm.tab10(i) for i in range(20)])
    for i, r in enumerate(tabEnt):
        tabEnt[r] = np.array(tabEnt[r])
        meanTrue, stdTrue = np.mean(tabEnt[r], axis=0), np.std(tabEnt[r], axis=0)

        ctmp = next(colors)
        plt.plot(tabNTot[r], meanTrue, color=ctmp, label=f"r={r}")
        plt.fill_between(tabNTot[r], meanTrue - stdTrue, meanTrue + stdTrue, color=ctmp, alpha=0.3)

    plt.legend()
    #plt.semilogy()
    #plt.semilogx()
    plt.savefig(folder+"Th_truc.pdf")
    plt.show()

def plotSumnkrvNrsumpk(folder, a):
    gdN = int(1e7)
    nbRuns = 100
    tabPopN={}
    tabPopNApprox={}
    tabNr = {}
    for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2]:
        print(r)
        tabPopN[r] = []
        tabPopNApprox[r] = []
        tabN = np.load(f"Data/DistribK/N_{r}_{a}_{gdN}_{0}.npy")
        tabNr[r] = tabN
        for run in range(nbRuns):
            with open(f"Data/DistribK/pop_{r}_{a}_{gdN}_{run}.txt", "r") as f:
                dicPopN = eval(f.read())

            tabPopTmp = []
            tabPopApproxTmp = []
            for i in range(0, len(tabN)):
                p = (np.array(dicPopN[tabN[i]][1:])**r)

                tabPopTmp.append(np.sum(p))
                #tabPopApproxTmp.append(tabN[i]**r*(np.sum(p/np.sum(p))))
                if r<=1:
                    arr = tabN[i]**((r**2+1)/2)
                else:
                    arr = tabN[i] ** (r)
                tabPopApproxTmp.append(arr)

            tabPopTmp = np.array(tabPopTmp)/tabPopTmp[-1]
            tabPopApproxTmp = np.array(tabPopApproxTmp)/tabPopApproxTmp[-1]
            tabPopN[r].append(tabPopTmp)
            tabPopNApprox[r].append(tabPopApproxTmp)

    colors = iter(reversed([plt.cm.tab20(i) for i in range(20)]*2))
    for i, r in enumerate(tabPopN):
        shift = 1./5**i
        meantrue = np.mean(tabPopN[r], axis=0) * shift
        meanApprox = np.mean(tabPopNApprox[r], axis=0) * shift
        stdtrue, stdApprox = np.std(tabPopN[r], axis=0) * shift, np.std(tabPopNApprox[r], axis=0) * shift
        c1 = next(colors)
        plt.plot(tabNr[r], meantrue, markersize=2, color=c1)
        plt.fill_between(tabNr[r], meantrue - stdtrue, meantrue + stdtrue, color=c1, alpha=0.3)
        plt.plot(tabNr[r], meanApprox, markersize=2, color=next(colors), label=f"r={r}")
    plt.xlabel(r"N")
    plt.ylabel(r"$\sum_k N_k^r$")
    plt.legend()
    plt.semilogx()
    plt.semilogy()
    plt.gca().set_yticklabels([])
    plt.savefig(folder+"Th2.pdf")
    #plt.show()
    plt.close()

def plotProbsvN(folder, a):
    gdN = int(1e7)
    for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2]:
        print(r)
        nbRuns = 1
        for run in range(nbRuns):
            with open(f"Data/DistribK/pop_{r}_{a}_{gdN}_{run}.txt", "r") as f:
                dicPopN = eval(f.read())

            tabPops = []
            tabNs = []
            tabKs = []
            tabN = []
            for n in dicPopN:
                popN = np.array(dicPopN[n][1:]) ** r
                popN /= np.sum(popN)

                for i, ni in enumerate(popN):
                    try:
                        tabPops[i].append(ni)
                        tabNs[i].append(n)
                    except:
                        tabPops.append([])
                        tabNs.append([])
                        tabPops[i].append(ni)
                        tabNs[i].append(n)
                tabKs.append(len(popN))
                tabN.append(n)

            for i in range(len(tabPops)):
                plt.scatter(tabNs[i], tabPops[i], s=1.)

            plt.ylim([1e-4, 1.05])
            plt.semilogy()
            plt.semilogx()
            plt.title(f"r={r}")
            plt.xlabel("N")
            plt.ylabel("Probabilities")
            arr = np.array(list(range(1, np.max(tabNs[-1]))))
            shift = 5
            if r>0.5:
                if r<1:
                    plt.plot(arr, arr**(-(r**2+1)/2)/shift,"--r", label=r"$N^{-\frac{r^2+1}{2}}$")
                else:
                    plt.plot(arr, arr**(-r)/shift,"--r", label=r"$N^{-r}$")

            else:
                plt.plot(tabN, 1./np.array(tabKs), "--k", label=r"$\frac{1}{K}$")
            plt.legend()

            plt.savefig(folder + f"Th1_r={r}.pdf")
            plt.savefig(folder + f"Th1_r={r}.png", dpi=300)
            #plt.show()
            plt.close()


r = 0.5
a = 1.
gdN = int(1e7)
folder = "Images/"

#plotKvN(folder, a)
#plotEntvr(folder, a)
#plotSumnkrvNrsumpk(folder, a)
#plotProbsvN(folder, a)
#pause()

import math
from math import log
from scipy.special import gammaln

def variation_of_information(X, Y):
  n = float(sum([len(x) for x in X]))
  sigma = 0.
  for x in X:
    p = len(x) / n
    for y in Y:
      q = len(y) / n
      r = len(set(x) & set(y)) / n

      if r > 0.0:
        sigma += - r * (log(r / p) + log(r / q))
  return abs(sigma)

def plotRes(results, name, c, norm=False, legend=""):
    arrx, arry, arryerr = [], [], []

    if name == "MargLik":
        resnorm = [np.mean(results[r][name]) for r in results]
        for r in results:
            print(r, np.mean(results[r][name]), np.min(resnorm))
            results[r][name] = results[r][name]-np.min(resnorm)

    maxval = 1.
    resnorm = [np.mean(results[r][name]) for r in results]
    if norm:
        maxval = np.max(resnorm)



    for r in results:
        arrx.append(r)
        arrres = np.array(results[r][name])

        arry.append(np.mean(results[r][name])/maxval)
        #arryerr.append(np.std(results[r][name])/maxval)
        arryerr.append(sem(results[r][name])/maxval)
    arry = np.array(arry)
    arryerr = np.array(arryerr)
    plt.plot(arrx, arry, color=c, label=legend)
    plt.fill_between(arrx, arry - arryerr, arry + arryerr, color=c, alpha=0.3)
    plt.xlabel("r")

def plotMetrics(folder, foldersave, typefig, small, shift):
    import os
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, v_measure_score, fowlkes_mallows_score

    toRem = []
    setFiles = os.listdir(folder)
    for i in range(len(setFiles)):
        if ".npy" not in setFiles[i] or setFiles[i][0] == "_" or setFiles[i][0] == ".":
            toRem.append(i)
        setFiles[i] = setFiles[i][:setFiles[i].rfind(".npy")]

    for i in reversed(toRem):
        del setFiles[i]
    setFiles = set(setFiles)

    results = {}
    colors = iter([plt.cm.tab10(i) for i in range(20)])
    for file in sorted(setFiles):
        type = file.split(" - ")[0]
        r = np.round(float(file.split(" - ")[1].replace("r=", "")), 3)
        shiftMeans = np.round(float(file.split(" - ")[2].replace("Shift=", "")), 1)
        alpha = np.round(float(file.split(" - ")[3].replace("Alpha=", "")), 2)
        fileindic = file.split(" - ")[4]
        i = int(file.split(" - ")[5].replace("run=", ""))

        if fileindic != "Data":
            continue

        if small:
            typefig+="_small"

        if r not in results:
            results[r] = {}
            results[r]["NMI"] = []
            results[r]["NVI"] = []
            results[r]["AdjRand"] = []
            results[r]["AdjMI"] = []
            results[r]["V-meas"] = []
            results[r]["Fowlkes"] = []
            results[r]["MargLik"] = []
            results[r]["varK"] = []

        if shiftMeans==shift and typefig.lower() == type.lower():
            print(file)
            try:
                logL = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - logL - run={i}.npy")
                means = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - means - run={i}.npy", allow_pickle=True)
                sigmas = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - sigmas - run={i}.npy", allow_pickle=True)
                tabK = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - K - run={i}.npy")
                X = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - Data - run={i}.npy")
                tabYTrue = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - YTrue - run={i}.npy")
                tabYInf = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - TInf - run={i}.npy")
            except Exception as e:
                print(e)
                continue

            partTrue, partInf = [], []
            for c in set(tabYTrue):
                partTrue.append(list(np.where(tabYTrue==c)[0]))
            for c in set(tabYInf):
                partInf.append(list(np.where(tabYInf==c))[0])

            maxVI = log(len(tabYTrue))
            NMI = normalized_mutual_info_score(tabYTrue, tabYInf)
            NVI = variation_of_information(partTrue,partInf) / maxVI
            AdjRand = adjusted_rand_score(tabYTrue, tabYInf)
            AdjMI = adjusted_mutual_info_score(tabYTrue,tabYInf)
            Vmeas = v_measure_score(tabYTrue, tabYInf)
            Fowlkes = fowlkes_mallows_score(tabYTrue, tabYInf)
            MargLik = logL
            varK = np.abs((len(set(tabYInf))-len(set(tabYTrue)))/len(set(tabYTrue)))

            if False:
                print(r, "\t",
                      NMI, "\t",
                      NVI, "\t",
                      AdjRand, "\t",
                      AdjMI, "\t",
                      Vmeas, "\t",
                      Fowlkes, "\t",
                      MargLik)

            results[r]["NMI"].append(NMI)
            results[r]["NVI"].append(NVI)
            results[r]["AdjRand"].append(AdjRand)
            results[r]["AdjMI"].append(AdjMI)
            results[r]["V-meas"].append(Vmeas)
            results[r]["Fowlkes"].append(Fowlkes)
            results[r]["MargLik"].append(MargLik)
            results[r]["varK"].append(varK)

    with open(f"Results/Results_{typefig}.txt", "w+") as f:
        for r in results:
            for name in results[r]:
                f.write(f"{r}\t{name}\t{np.mean(results[r][name])}\t{sem(results[r][name])}\n")
    plt.figure(figsize=(4,4))
    #plotRes(results, "NMI", c=next(colors), legend="NMI")
    plotRes(results, "AdjMI", c=next(colors), legend="Adj. MI")
    plotRes(results, "AdjRand", c=next(colors), legend="Adj. rand index")
    plotRes(results, "NVI", c=next(colors), legend="Norm. VI")
    #plotRes(results, "V-meas", c=next(colors), legend="V measure")
    plotRes(results, "Fowlkes", c=next(colors), legend="Fowlkes-Mallows score")
    plt.plot([1, 1], [0, 1], "--k")
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    sm=""
    if small: sm="_small"
    plt.savefig(foldersave+f"/Metrics {typefig}{sm}_shift={shift}.pdf", dpi=600)
    #plt.show()
    plt.close()

    plt.figure(figsize=(4,4))
    plotRes(results, "varK", c=next(colors), legend=r"$\frac{K_{inf} - K_{true}}{K_{true}}$")
    plotRes(results, "MargLik", c=next(colors), legend="Norm. log-likelihood", norm=True)
    plt.plot([1, 1], [0, np.max([1, np.max(results[1.]["varK"])])], "--k")
    plt.ylim([0,1])
    plt.legend()
    plt.tight_layout()
    sm=""
    if small: sm="_small"
    plt.savefig(foldersave+f"/Metrics2 {typefig}{sm}_shift={shift}.pdf", dpi=600)
    #plt.show()
    plt.close()

def plotHeatmap(folder, foldersave, typefig, small):
    import os
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, v_measure_score, fowlkes_mallows_score

    toRem = []
    setFiles = os.listdir(folder)
    for i in range(len(setFiles)):
        if ".npy" not in setFiles[i] or setFiles[i][0] == "_" or setFiles[i][0] == ".":
            toRem.append(i)
        setFiles[i] = setFiles[i][:setFiles[i].rfind(".npy")]

    for i in reversed(toRem):
        del setFiles[i]
    setFiles = set(setFiles)

    results = {}
    for file in sorted(setFiles):
        type = file.split(" - ")[0]
        r = np.round(float(file.split(" - ")[1].replace("r=", "")), 3)
        shiftMeans = np.round(float(file.split(" - ")[2].replace("Shift=", "")), 1)
        alpha = np.round(float(file.split(" - ")[3].replace("Alpha=", "")), 2)
        i = int(file.split(" - ")[5].replace("run=", ""))

        if r not in results:
            results[r] = {}
        if shiftMeans not in results[r]:
            results[r][shiftMeans]={}
            results[r][shiftMeans]["NMI"] = []
            results[r][shiftMeans]["NVI"] = []
            results[r][shiftMeans]["AdjRand"] = []
            results[r][shiftMeans]["AdjMI"] = []
            results[r][shiftMeans]["V-meas"] = []
            results[r][shiftMeans]["Fowlkes"] = []
            results[r][shiftMeans]["MargLik"] = []

        if typefig.lower() in type.lower() and (("small" not in type.lower() and not small) or ("small" in type.lower() and small)):
            print(file)
            try:
                logL = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - logL - run={i}.npy")
                means = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - means - run={i}.npy", allow_pickle=True)
                sigmas = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - sigmas - run={i}.npy", allow_pickle=True)
                tabK = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - K - run={i}.npy")
                X = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - Data - run={i}.npy")
                tabYTrue = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - YTrue - run={i}.npy")
                tabYInf = np.load(folder + type + " - "  + f"r={r} - Shift={shiftMeans} - Alpha={alpha} - TInf - run={i}.npy")
            except Exception as e:
                print(e)
                continue

            partTrue, partInf = [], []
            for c in set(tabYTrue):
                partTrue.append(list(np.where(tabYTrue==c)[0]))
            for c in set(tabYInf):
                partInf.append(list(np.where(tabYInf==c))[0])

            maxVI = log(len(tabYTrue))
            NMI = normalized_mutual_info_score(tabYTrue, tabYInf)
            NVI = variation_of_information(partTrue,partInf) / maxVI
            AdjRand = adjusted_rand_score(tabYTrue, tabYInf)
            AdjMI = adjusted_mutual_info_score(tabYTrue,tabYInf)
            Vmeas = v_measure_score(tabYTrue, tabYInf)
            Fowlkes = fowlkes_mallows_score(tabYTrue, tabYInf)
            MargLik = logL

            if False:
                print(r, "\t",
                      NMI, "\t",
                      NVI, "\t",
                      AdjRand, "\t",
                      AdjMI, "\t",
                      Vmeas, "\t",
                      Fowlkes, "\t",
                      MargLik)

            results[r][shiftMeans]["NMI"].append(NMI)
            results[r][shiftMeans]["NVI"].append(NVI)
            results[r][shiftMeans]["AdjRand"].append(AdjRand)
            results[r][shiftMeans]["AdjMI"].append(AdjMI)
            results[r][shiftMeans]["V-meas"].append(Vmeas)
            results[r][shiftMeans]["Fowlkes"].append(Fowlkes)
            results[r][shiftMeans]["MargLik"].append(MargLik)

    def heat(foldersave, results, name):
        mat = []
        arrr = sorted(results.keys())
        for r in arrr:
            mat.append([])
            arrs = sorted(results[r].keys())
            for shift in arrs:
                mat[-1].append(np.mean(results[r][shift][name]) - np.mean(results[1.][shift][name]))
        mat = np.array(mat)

        sns.heatmap(mat, cmap="RdBu_r", center=0)
        plt.gca().set_xticklabels(arrs)
        plt.xlabel("shift")
        plt.gca().set_yticklabels(arrr, rotation=0)
        plt.ylabel("r")

        sm = ""
        if small: sm = "_small"
        plt.title(typefig)
        plt.savefig(foldersave + f"/Heatmap {typefig}{sm}_{name}.pdf")
        #plt.show()
        plt.close()

    heat(foldersave, results, "NMI")
    heat(foldersave, results, "NVI")
    heat(foldersave, results, "AdjMI")
    heat(foldersave, results, "AdjRand")
    heat(foldersave, results, "V-meas")
    heat(foldersave, results, "Fowlkes")
    heat(foldersave, results, "MargLik")


folder = "Data/XP/"
foldersave = "Results/"
alpha = 1.
small = False

for typefig in ["Grid", "Diamonds", "Density"]:
    #plotHeatmap(folder, foldersave, typefig, small)
    for shift in [1.]:
        plotMetrics(folder=folder, foldersave=foldersave, typefig=typefig, small=small, shift=shift)

        continue
        sm = ""
        if small: sm = "_small"
        X = np.load(folder + typefig + sm + " - " + f"r={1.} - Shift={shift} - Alpha={alpha} - Data - run=0.npy")
        plt.scatter(X[:, 0], X[:, 1], s=2)
        plt.gca().set_aspect("equal")
        plt.savefig(foldersave + f"/Data {typefig}{sm}_shift={shift}.pdf", dpi=600)
        #plt.show()
        plt.close()













