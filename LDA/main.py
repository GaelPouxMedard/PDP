import numpy as np
import time
import codecs
import jieba
import re
import os
import sys


def prepreprocessing():
    textFinal = ""
    folderstrain = os.listdir("20news-bydate-train")
    dicIndex = {}
    indexDoc = 0
    for folder in folderstrain:
        files = os.listdir("20news-bydate-train/" + folder)
        for file in files:
            dicIndex[file] = []
            with open("20news-bydate-train/" + folder + "/" + file, "r") as f:
                text = f.read()
                text = text[text.find("\n"):]
                lines = text.split("\n")
                for line in lines:
                    line = " ".join(line.split())
                    line = line.replace("\n", "")
                    if len(line.split(" ")) < 3: continue
                    if len(line) < 10: continue
                    textFinal += line + " "
                    dicIndex[file].append(indexDoc)
                    indexDoc += 1
                textFinal += "\n"
    print(textFinal)

    with open("Dataset_20new.txt", "w+") as o:
        o.write(textFinal)

    with open("indexDoc.txt", "w+") as f:
        for key, val in dicIndex.items():
            f.write(f"{key}\t{val}\n")
    sys.exit()


def preprocessing(dataset):
    # 读取停止词文件
    file = codecs.open('stopwords.dic', 'r', 'utf-8')
    stopwords = [line.strip() for line in file]
    file.close()

    # 读数据集
    try:
        file = codecs.open(dataset, 'r')  # ,'utf-8')
        documents = [document.strip() for document in file]
    except:
        file = codecs.open(dataset, 'r', 'ISO-8859-1')
        documents = [document.strip() for document in file]

    file.close()

    word2id = {}
    id2word = {}
    docs = []
    currentDocument = []
    currentWordId = 0

    lg = len(documents)
    for index_doc, document in enumerate(documents):
        if index_doc % (lg // 10) == 0: print(index_doc * 100 / lg, "%")
        if index_doc > 350 and True: break
        # 分词
        segList = jieba.cut(document)
        for word in segList:
            word = word.lower().strip()
            # 单词长度大于1并且不包含数字并且不是停止词
            if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:
                if word in word2id:
                    currentDocument.append(word2id[word])
                else:
                    currentDocument.append(currentWordId)
                    word2id[word] = currentWordId
                    id2word[currentWordId] = word
                    currentWordId += 1
        docs.append(currentDocument);
        currentDocument = []
    return docs, word2id, id2word


def randomInitialize():
    for d, doc in enumerate(docs):
        zCurrentDoc = []
        for w in doc:
            pz = np.divide(np.multiply(ndz[d, :], nzw[:, w]), nz)
            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            zCurrentDoc.append(z)
            ndz[d, z] += 1
            nzw[z, w] += 1
            nz[z] += 1
        Z.append(zCurrentDoc)


def gibbsSampling(r):
    for d, doc in enumerate(docs):
        for index, w in enumerate(doc):
            z = Z[d][index]

            ndz[d, z] -= 1
            nzw[z, w] -= 1
            nz[z] -= 1

            pz = np.divide(np.multiply(ndz[d, :], nzw[:, w]), nz)
            pz = pz ** r

            z = np.random.multinomial(1, pz / pz.sum()).argmax()
            Z[d][index] = z

            ndz[d, z] += 1
            nzw[z, w] += 1
            nz[z] += 1


def perplexity():
    nd = np.sum(ndz, 1)
    n = 0
    ll = 0.0
    for d, doc in enumerate(docs):
        for w in doc:
            ll = ll + np.log(((nzw[:, w] / nz) * (ndz[d, :] / nd[d])).sum())
            n = n + 1
    return np.exp(ll / (-n))


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


r = 0.5

alpha = 1.
beta = 0.1
iterationNum = 200
nbRuns = 100
K = 20
dataset = "Dataset_20new.txt"
# dataset = "dataset.txt"

# prepreprocessing()
docs, word2id, id2word = preprocessing(dataset)

N = len(docs)
M = len(word2id)
print(N, M)

tabLabs = ["NMI", "NVI", "AdjRand", "AdjMI", "Vmeas", "Fowlkes", "MargLik", "varK"]
with open(f"Results/Results_{r}.txt", "w+") as f:
    for lab in tabLabs:
        f.write(str(lab) + "\t")
    f.write("\n")

nameClusToIndex = {}
ind_clus = 0
tabYTrue = np.zeros((N))
with open("indexDoc.txt", "r") as f:
    for line in f:
        nameClus, docsInIndex = line.replace("\n", "").split("\t")
        if nameClus not in nameClusToIndex:
            nameClusToIndex[nameClus] = ind_clus
            ind_clus += 1

        docsInIndex = docsInIndex.replace("[", "").replace("]", "").split(", ")
        for doc in docsInIndex:
            try:  # Only when not all the dataset is considered
                tabYTrue[int(doc)] = nameClusToIndex[nameClus]
            except:
                pass

for run in range(nbRuns):
    Z = []
    ndz = np.zeros([N, K]) + alpha
    nzw = np.zeros([K, M]) + beta
    nz = np.zeros([K]) + M * beta
    randomInitialize()

    tabYInf = np.zeros((N))

    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score, \
        v_measure_score, fowlkes_mallows_score

    t = time.time()
    for i in range(0, iterationNum):
        gibbsSampling(r)
        dt = time.time()-t
        t = time.time()
        print(dt, "s", f"Iteration: {i}/{iterationNum} Completed", " Perplexity: ", perplexity())

        if i%10==0:
            for doc_i, z in enumerate(Z):
                ind, cnt = np.unique(z, return_counts=True)
                topic = ind[cnt == max(cnt)][0]
                tabYInf[doc_i] = topic

            partTrue, partInf = [], []
            for c in set(tabYTrue):
                partTrue.append(list(np.where(tabYTrue == c)[0]))
            for c in set(tabYInf):
                partInf.append(list(np.where(tabYInf == c))[0])
            maxVI = np.log(len(tabYTrue))
            NMI = normalized_mutual_info_score(tabYTrue, tabYInf)
            NVI = variation_of_information(partTrue, partInf) / maxVI
            AdjRand = adjusted_rand_score(tabYTrue, tabYInf)
            AdjMI = adjusted_mutual_info_score(tabYTrue, tabYInf)
            Vmeas = v_measure_score(tabYTrue, tabYInf)
            Fowlkes = fowlkes_mallows_score(tabYTrue, tabYInf)
            MargLik = perplexity()
            varK = np.abs((len(set(tabYInf)) - len(set(tabYTrue))) / len(set(tabYTrue)))
            tabMetrics = [NMI, NVI, AdjRand, AdjMI, Vmeas, Fowlkes, MargLik, varK]
            with open(f"Results/Results_{r}.txt", "a") as f:
                for met in tabMetrics:
                    f.write(str(met) + "\t")
                f.write("\n")
            print("\t".join(tabLabs))
            print("\t".join(list(map(str, tabMetrics))))
            print()
