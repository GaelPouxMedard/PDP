import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from copy import deepcopy as copy
from math import gamma
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'

def beta(x, shift=0.):
    return np.prod([gamma(c+shift) for c in x])/gamma(np.sum(x)+shift)

plt.figure()

def drawBranchs(cnt, endPoints, fin, depth):
    newEndPoints = []
    newCnt = []
    tabProbs = []


    for i in range(len(endPoints)):
        for k in range(K):
            cntTmp = copy(cnt[i])

            angle = (np.pi/4)/(depth**2+1)

            p = np.product(probs[k] ** -((cntTmp[k]+1)**r-cntTmp[k]**r))# / np.product(probs[k] ** (cntTmp[k]))
            p /= np.sum(probs ** -((cntTmp+1)**r-cntTmp**r))

            p = ((cntTmp[k]+1)**r + probs[k])/np.sum((cntTmp+1)**r + probs)
            sign = ((k-0.5) / (0.5))
            shifty = sign * p * angle
            shiftx = (p**2 - shifty**2) ** (0.5)

            print(cntTmp, p, (cntTmp[k]+1)**r, np.sum((cntTmp+1)**r))


            plt.plot([endPoints[i][0], endPoints[i][0] + shiftx], [endPoints[i][1], endPoints[i][1] + shifty], "-k")
            plt.plot([endPoints[i][0], endPoints[i][0] + shiftx], [endPoints[i][1], endPoints[i][1] + shifty], "or", markersize=2)

            cntTmp[k]+=1
            newEndPoints.append([endPoints[i][0]+shiftx, endPoints[i][1] + shifty])
            newCnt.append(cntTmp)
            tabProbs.append(p)

    return newEndPoints, newCnt, tabProbs

K = 2
p = [0.5,0.5]
ifin = 5
r = 1.5
probs = np.zeros((K))+np.array(p)
probs /= np.sum(probs)


endPoints = [[1., 1.]]
orshift = copy(endPoints)
cnts = [np.zeros((K))]
fin = False
for i in range(ifin):
    print("Depth", i)
    newEndPoints = []
    newCnts = []
    if i==ifin-1: fin=True

    newEndPoint, newCnt, tabProbs = drawBranchs(cnts, endPoints, fin, i)
    newEndPoints+=newEndPoint
    newCnts+=newCnt

    endPoints = copy(newEndPoints)
    cnts = copy(newCnts)

if False:
    print(endPoints)
    endPoints = np.array(endPoints)
    endPoints[:, 0] = np.max(endPoints[:, 0])*1.02
    maxP = 0.
    for i in range(len(endPoints)):
        div = np.sum(probs ** ((cnts[i]+1) ** r-cnts[i]**r))
        p = 1.
        for k in range(K):
            cntTmp = copy(cnts[i])
            cntTmp[k] += 1

            #p = np.product(probs ** (cntTmp) ** r)/div  # / np.product(probs[k] ** (cntTmp[k]))
            p *= cntTmp ** r / div
        if p>maxP: maxP = p
    maxP *= 2
    for i in range(len(endPoints)):
        div = np.sum(probs ** ((cnts[i] + 1) ** r - cnts[i] ** r))
        for k in range(K):
            cntTmp = copy(cnts[i])
            cntTmp[k] += 1

            angle = 0.
            p = np.product(probs ** (cntTmp) ** r) / div  # / np.product(probs[k] ** (cntTmp[k]))
            p /= maxP
            print(p, cntTmp)
            # p = np.log(p)
            sign = ((k - 0.5) / (0.5))
            shifty = sign * p * angle
            shiftx = (p ** 2 - shifty ** 2) ** (0.5)

            plt.plot([endPoints[i][0], endPoints[i][0] + shiftx], [endPoints[i][1], endPoints[i][1] + shifty], "-k")
            #plt.plot([endPoints[i][0], endPoints[i][0] + shiftx], [endPoints[i][1], endPoints[i][1] + shifty], "or", markersize=2)



plt.gca().set_aspect("equal")
#plt.savefig(f"Tree_r={r}.pdf", dpi=300)
plt.show()