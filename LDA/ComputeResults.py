import os
import numpy as np
from scipy.stats import sem


listRes = os.listdir("Results")

fileoutputmean = open("Results_mean.txt", "a+")
fileoutputstd = open("Results_std.txt", "a+")
fileoutputsem = open("Results_sem.txt", "a+")
fileoutputmean.truncate(0)
fileoutputstd.truncate(0)
fileoutputsem.truncate(0)

first = True
for file in listRes:
    r = float(file[-7:-4])

    with open("Results/"+file, "r") as f:
        labels = f"{r}\t"+f.readline()
        if first:
            fileoutputmean.write(labels)
            fileoutputstd.write(labels)
            fileoutputsem.write(labels)
            first=False
        tabRes = []
        for line in f:
            line = line.replace("\n", "").split("\t")
            tabRes.append(list(map(float, line)))

        mean = np.mean(tabRes, axis=1)
        std = np.std(tabRes, axis=1)
        sem = sem(tabRes, axis=1)

        fileoutputmean.write(f"{r}\t{'\t'.join(mean)}\n")
        fileoutputstd.write(f"{r}\t{'\t'.join(std)}\n")
        fileoutputsem.write(f"{r}\t{'\t'.join(sem)}\n")

fileoutputmean.close()
fileoutputstd.close()
fileoutputsem.close()