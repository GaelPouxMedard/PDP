from scipy.special import hyp2f1
import matplotlib.pyplot as plt
import numpy as np
from math import gamma
from itertools import product, permutations
import time
from copy import deepcopy as copy

def fac(x):
    return gamma(x+1)
def beta(x, shift=0.):
    return np.prod([gamma(c+shift) for c in x])/gamma(np.sum(x)+shift)
def dirichlet(a, p):
    return np.prod(p**(a))/beta(a)
def H(arr, r):
    sp = np.linspace(1, np.max(arr)+1, np.max(arr)+1, dtype=int)


    a2 = 1./(sp**r)
    a2 = np.cumsum(a2)
    return a2[arr-1]

    from scipy.special import zeta
    return zeta(r, 1) - zeta(r, arr+1)


#np.random.seed(11111)


if False:
    # ======================= POWERED MULTINOMIAL =======================
    K = 2
    N = 2
    r = 2

    permut = list(product(list(range(K)), repeat=N))
    probs = np.random.random((K))
    probs /= np.sum(probs)
    alpha = np.zeros((K))+1

    #probs = np.ones((K)) / K

    permutVues = []
    permutNew = []
    for p in permut:
        if p not in permutVues:
            vu = False
            for pvues in permutVues:
                if p in pvues:
                    vu = True

            if not vu:
                permutVues.append(list(permutations(p, len(p))))
                permutNew.append(p)
    permut = permutNew  # Fixed in the marble
    print(permut)

    arrTot = []
    div = 0.
    div1, div2 = 0., 0.
    dicDiv = {}
    sumVecCntr = 0.
    arrSumVecCnts = np.zeros((K))
    arrVecCnts = []
    for perm in permut:
        perm = np.array(perm)
        vecCnt = np.zeros((K))

        for ind in set(perm):
            cnt = float(len(perm[perm == ind]))
            vecCnt[ind] = cnt
        vecCnt = np.array(vecCnt)

        Nhyp = np.sum(vecCnt**r)
        if Nhyp not in dicDiv:dicDiv[Nhyp] = 0.

        nbChemNorm = 1/(beta(vecCnt, shift=1))
        nbChemHyp = 1/beta(vecCnt**r, shift=1)
        nbChemNormTot = K**(np.sum(vecCnt))
        nbChemHypTot = K**(np.sum(vecCnt**r))

        tmp = np.product(probs**(vecCnt**r))
        tmp *= nbChemNorm

        print(vecCnt, nbChemNorm, nbChemHyp)
        div1 += nbChemNorm
        div2 += nbChemHyp
        sumVecCntr += np.sum(vecCnt**r)#*nbChemNorm/nbChemHyp
        arrSumVecCnts += vecCnt**r #*nbChemNorm/nbChemHyp
        arrVecCnts.append(vecCnt**r)

        arrTot.append(tmp)

    from scipy.stats import multinomial
    m=multinomial(sumVecCntr, probs)

    div = m.pmf(arrSumVecCnts)

    print()
    print(arrTot[1]/arrTot[0])
    print(arrTot, np.sum(arrTot), div, div1/div2)
    tot = np.sum(arrTot) / div

    print(tot)

    time.sleep(0.1)
    pause()




def runPCRP(gdN, a, r):
    K = 1
    inds = np.array(np.logspace(0, np.log10(gdN), int(np.log(gdN) * 100)), dtype=int)
    popK = np.array([a])
    tabN, tabK, dicPopK = [], [], {}
    for n in range(gdN):
        if n % (gdN / 100) == 0 and False:
            #print(n * 100 / gdN, "%")
            if K>=5:
                print(popK[1]/popK[0], popK[1]/popK[2], popK[1]/popK[3], popK[1]/popK[4], list(popK/np.sum(popK))[:5])
        cs = np.cumsum(popK[1:] ** r)
        cs = np.append(np.array([a]), cs+a)
        rand = np.random.random() * cs[-1]
        k = np.argmax(cs > rand)

        if k == 0:
            popK = np.append(popK, 1)
            K += 1
        else:
            popK[k] += 1

        if n in inds:
            tabN.append(n)
            tabK.append(K)
            dicPopK[n]=list(popK)

    return tabN, tabK, dicPopK

r = 0.3
a = 1.
gdN = 1e7
gdN = int(gdN)
a = float(a)

loadRes = False
nbRuns = 100
if not loadRes:
    for run in range(nbRuns):
        print("r =", r, "RUN -", run)
        tabN, tabK, dicPopN = runPCRP(gdN, a, r)

        np.save(f"Data/DistribK/N_{r}_{a}_{gdN}_{run}", tabN)
        np.save(f"Data/DistribK/K_{r}_{a}_{gdN}_{run}", tabK)
        with open(f"Data/DistribK/pop_{r}_{a}_{gdN}_{run}.txt", "w+") as f: f.write(str(dicPopN))

    tabK = np.array(tabK)
    tabN = np.array(tabN)

else:
    tabNTot = None
    tabKTot = None
    for run in range(nbRuns):
        print(run)
        tabN = np.load(f"Data/DistribK/N_{r}_{a}_{gdN}_{run}.npy")
        tabK = np.load(f"Data/DistribK/K_{r}_{a}_{gdN}_{run}.npy")

        try:
            ind = np.array(np.logspace(0, np.log10(np.max(tabN)), int(np.log(np.max(tabN))*100)), dtype=int)[:-1]
            tabK = np.array(tabK)[ind]
            tabN = np.array(tabN)[ind]
        except:
            print("REDO EXP WITH POP-N")

        tabK = np.array(tabK)
        tabN = np.array(tabN)

        if tabNTot is None:
            tabNTot = np.array(copy(tabN))
            tabKTot = np.array(copy(tabK))
        else:
            #tabNTot+=tabN
            tabKTot+=tabK


    tabKTot = tabKTot / nbRuns
    tabK = tabKTot


arr = tabN
try:
    with open(f"Data/DistribK/pop_{r}_{a}_{gdN}_{run}.txt", "r") as f:
        dicPopN = eval(f.read())
    popN = np.array(dicPopN[tabN[-1]])
    div = np.sum((popN[1:] / np.sum(popN[1:])) ** r)
    print(div)

    if False:
        tabprobs = np.array(list(sorted(popN, reverse=True)))
        sortedpop = np.array(list(sorted(popN, reverse=True)))
        tabprobs /= np.sum(tabprobs)
        sortedpop = sortedpop[sortedpop<100]
        plt.plot(range(len(tabprobs)), tabprobs)

        x=np.linspace(0, len(tabprobs), 100)
        y = np.max(tabprobs)-x**(-r) * np.max(tabprobs)/np.max(x**(1-r))
        plt.plot(x, y)
        plt.show()
        plt.close()
except Exception as e:
    print(e)
    print("REDO EXP WITH POP-N")
    div = 1.

#arr=np.linspace(1, max(tabN), max(tabN))
s=0
arrData = np.array(tabK)
arr1 = a*np.log(arr)
arr2 = a*((arr**(1-r)-1)/((1-r)*div) + arr**((1-r)/2))
arr3 = a*H(arr, r)/div
arr4 = a*arr**((1-r)/2)
arr5 = a*(arr**(1-r)-1)/((1-r)*div)
arr6 = (2*a)**0.5 * arr**0.5

#print(arr3)

#arrData = arrData/arrData[-1];arr1 /= arr1[-1];arr2 /= arr2[-1];arr3 /= arr3[-1];arr4 /= arr4[-1];arr5 /= arr5[-1];arr6 /= arr6[-1]
#plt.plot(arr, np.sqrt(arr), label="sqrt")
plt.plot(arr, arr1, label="log")
plt.plot(arr, arr2, label="Compo")
plt.plot(arr, arr3, label="H(N,r)")
plt.plot(arr, arr4, label="Approx gen")
plt.plot(arr, arr5, label="Approx r=1")
plt.plot(arr, arr6, label="Approx r=0")

plt.plot(tabN, arrData, label="data")
plt.legend()
plt.title(str(r)+" - "+str(a))
plt.semilogy()
#plt.semilogx()
plt.show()
pause()




def testDistribs():
    def fac(x):
        return gamma(x + 1)

    def beta(x, shift=0.):
        return np.prod([gamma(c + shift) for c in x]) / gamma(np.sum(x) + shift)
    # ======================= POWERED DIRICHLET-MULTINOMIAL =======================
    from scipy.stats import dirichlet
    from itertools import product, permutations, combinations

    K = 2
    N = 4
    r = 1.
    alpha = np.ones((K))
    a2 = np.random.random((K))

    permut = list(product(list(range(K)), repeat=N))
    probs = np.random.random((K))
    probs /= np.sum(probs)

    probs = np.ones((K)) / K

    permutVues = []
    permutNew = []
    for p in permut:
        if p not in permutVues:
            vu = False
            for pvues in permutVues:
                if p in pvues:
                    vu = True

            if not vu:
                permutVues.append(list(permutations(p, len(p))))
                permutNew.append(p)
    permut = permutNew  # Fixed in the marble
    print(permut)
    arrTot = []
    div = 0
    for perm in permut:

        perm = np.array(perm)
        cnt = []
        for k in range(K):
            cnt.append(len(perm[perm == k]))
        cnt = np.array(cnt)

        for perm2 in permut:

            perm2 = np.array(perm2)
            cnt2 = []
            for k in range(K):
                cnt2.append(len(perm2[perm2 == k]))
            cnt2 = np.array(cnt2)


        tmp = beta(alpha + cnt ** r)

        tmp /= beta(cnt, shift=1)  # Pour pas considérer de permutations

        arrTot.append(tmp)
        div += tmp

    print(arrTot)
    tot = np.sum(arrTot)
    print(tot / div)
    print(tot)

    pause()

    # ======================= POWERED MULTINOMIAL =======================
    K = 2
    N = 3
    r = 2

    permut = list(product(list(range(K)), repeat=N))
    probs = np.random.random((K))
    probs /= np.sum(probs)

    probs = np.ones((K)) / K

    permutVues = []
    permutNew = []
    for p in permut:
        if p not in permutVues:
            vu = False
            for pvues in permutVues:
                if p in pvues:
                    vu = True

            if not vu:
                permutVues.append(list(permutations(p, len(p))))
                permutNew.append(p)
    permut = permutNew  # Fixed in the marble
    print(permut)

    arrTot = []
    div = 0.
    for perm in permut:
        x1 = 0.7
        # probs = np.array([x1, (1-x1**r)**(1/r)])

        perm = np.array(perm)
        tmp = 1.
        vecCnt = np.zeros((K))
        for ind in set(perm):
            cnt = float(len(perm[perm == ind]))

            tmp *= (probs[ind] ** (cnt ** r))
            vecCnt[ind] = cnt

        vecCnt = np.array(vecCnt)
        tmp /= (beta((vecCnt), shift=1))

        print(1. / (beta((vecCnt), shift=1)), tmp, vecCnt, np.product(vecCnt ** r))

        div += np.prod((probs ** (vecCnt ** r))) / (beta((vecCnt), shift=1))
        # tmp /= 1*beta(vecCnt, shift=1)

        # tmp *= fac(N**r)
        # div += np.sum((beta((vecCnt)**(r), shift=1))**r)

        arrTot.append(tmp)

    print(arrTot)
    tot = np.sum(arrTot) / div

    print(tot)

    pause()

    # ======================= POWERED DIRICHLET (sum p**r = 1) =======================
    K = 2
    N = 6

    taba = np.zeros((K)) + 20.
    r = 0.9

    sp = np.linspace(0., 1., 10000)
    step = 1. / len(sp)
    tot = 0.
    cst = gamma(np.sum((taba))) * r ** (K - 1) / np.product([gamma(a) for a in taba])

    norm = gamma(r * np.sum((taba))) * np.product([gamma(a) for a in taba]) * r ** (-K)
    norm /= gamma(np.sum((taba))) * np.product([gamma(r * a) for a in taba])
    norm = 1.
    for x1 in sp:
        x2 = (1 - x1 ** r) ** (1 / r)
        # x2 = 1-x1

        tmp = x1 ** ((taba[0]) * r - 1) * x2 ** ((taba[1]) * r - 1) * step

        tmp *= x2 ** (1. - r)

        print(tmp, x1 ** r + x2 ** r)

        # print(tmp*cst*norm, x1**r+x2**r)

        if np.isfinite(tmp):
            tot += tmp
        else:
            print("ERR")

    tot *= cst * norm

    print(tot)
    pause()