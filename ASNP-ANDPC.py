# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 20:35:26 2021

@author: 35119
"""

import numpy as np
import matplotlib.pyplot as plt
import csv  # 可以导入csv文件
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from munkres import Munkres
import math
from scipy.spatial.distance import pdist
# 计算数据点两两之间的距离


def getDistanceMatrix(newdatas):
    N, D = np.shape(newdatas)
    dists = np.zeros([N, N])
    #list=[]
    for i in range(N):
        for j in range(N):
            vi = newdatas[i, :]
            vj = newdatas[j, :]
            if i != j:
                dists[i, j] =np.sqrt(np.dot((vi - vj), (vi - vj)))
            else:
                dists[i,i]=float("inf")
            #A = np.log((vi+vj)/2)
            #B = (np.log(vi)+np.log(vj))/2
            #dists[i,j]=np.sum(A-B)
    return dists

def kneigbour(dists):
    N=np.shape(dists)[0]
    nn = np.zeros([N,N])
    for i in range (len(dists)):
        nn[i:] = np.sort(dists[i,:])
    return nn
def kvalue(dists):
    N = np.shape(dists)[0]
    NND = np.zeros(N)
    for i in range(len(dists)):
        NND[i] = np.min(dists[i, :])
    MNND = np.max(NND)/2
    print(MNND)
    condlist=[nn<=MNND]
    print(condlist)
    choicelist=[nn+0]
    nnn=np.select(condlist,choicelist,default=0)
    print(nnn)
    return nnn
# 计算每个点的局部密度
def get_density(nnn):
    N = np.shape(dists)[0]
    rho = np.zeros(N)
    exist=(nnn>0)*1.0
    factor=np.ones(nnn.shape[1])
    res=np.dot(exist,factor)
    print(res)
    num = np.sum(res)
    k=int(num/N)
    #k=int(N*p)
    #k=math.ceil(N*p)
    kneigh = nn[:,:k]

    for i in range(N):
        rho[i] = np.exp((-1/k)*(np.sum((kneigh[i,:]) ** 2)))
        #rho[i] = np.exp((-1/(res[i]))*(np.sum(nnn[i, :]**2)))
    return rho


# 计算每个数据点的密度距离
# 即对每个点，找到密度比它大的所有点
# 再在这些点中找到距离其最近的点的距离
def get_deltas(dists, rho):
    N = np.shape(dists)[0]
    deltas = np.zeros(N)
    nearest_neiber = np.zeros(N)
    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 对于密度最大的点
        if i == 0:
            continue
        # 对于其他的点
        # 找到密度比其大的点的序号
        index_higher_rho = index_rho[:i]
        # 获取这些点距离当前点的距离,并找最小值
        deltas[index] = np.min(dists[index, index_higher_rho])

        # 保存最近邻点的编号
        index_nn = np.argmin(dists[index, index_higher_rho])
        nearest_neiber[index] = index_higher_rho[index_nn].astype(int)

    deltas[index_rho[0]] = np.max(deltas)
    return deltas, nearest_neiber


# 选取 rho与delta乘积较大的点作为
# 聚类中心

def find_centers_K(rho, deltas, K):
    rho_delta = rho * deltas
    centers = np.argsort(-rho_delta)
    return centers[:K]


def cluster_PD(rho, centers, nearest_neiber):
    K = np.shape(centers)[0]
    if K == 0:
        print("can not find centers")
        return

    N = np.shape(rho)[0]
    labs = -1 * np.ones(N).astype(int)

    # 首先对几个聚类中进行标号
    for i, center in enumerate(centers):
        labs[center] = i

    # 将密度从大到小排序
    index_rho = np.argsort(-rho)
    for i, index in enumerate(index_rho):
        # 从密度大的点进行标号
        if labs[index] == -1:
            # 如果没有被标记过
            # 那么聚类标号与距离其最近且密度比其大
            # 的点的标号相同
            labs[index] = labs[int(nearest_neiber[index])]
    return labs

def best_map(y, y_pred):
    # L1 should be the labels and L2 should be the clustering number we got
    y_pred=y_pred+1
    Label1 = np.unique(y)  # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)  # 标签的大小
    Label2 = np.unique(y_pred)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pred == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    #c = index[:, 1]
    index=index+1
    newy_pred = np.zeros(y_pred.shape)
    for i in range(nClass2):
        for j in range(len(y_pred)):
            if y_pred[j] == index[i, 0]:
                newy_pred[j] = index[i, 1]
        #newy_pred[y_pred == Label2[i]] = Label1[c[i]]
    return newy_pred




if __name__ == "__main__":

    file_name='ecoli'#没有第一行且标签从0开始
    with open('ecoli.csv', 'r') as fc:
        reader = csv.reader(fc)
        #next(reader)
        labels_true1 = []
        lines1 = []
        for line in reader:
            lines1.append(line[:-1])
            labels_true1.append(line[-1])
    lines = lines1
    #print("lines:\n", lines)
    labels_true = np.array(labels_true1, dtype=int)
    #print(labels_true)
    datas = np.array(lines).astype(np.float64)
    #print("datas:\n", datas)
    #datas =Normalizer(norm="l2").fit_transform(datas)

    #pca = PCA(n_components=7)
    #datas= pca.fit_transform(datas)

    # 计算距离矩阵
    dists = getDistanceMatrix(datas)
    nn = kneigbour(dists)
    nnn=kvalue(dists)
    # 计算局部密度
    rho = get_density(nnn)
    # 计算密度距离
    deltas, nearest_neiber = get_deltas(dists, rho)

    # 绘制密度/距离分布图
    #draw_decision(rho, deltas, name=file_name + "_decision.jpg")

    # 获取聚类中心点
    centers = find_centers_K(rho, deltas, 8)

    # centers = find_centers_auto(rho,deltas)
    print("centers", centers)

    labs = cluster_PD(rho, centers, nearest_neiber)
    newy_labs = best_map(labels_true, labs)

    print(metrics.adjusted_rand_score(labels_true, newy_labs))
    # print(metrics.adjusted_mutual_info_score(labels_true,labs))
    print(metrics.normalized_mutual_info_score(labels_true, newy_labs))
    print(metrics.f1_score(labels_true, newy_labs, average='weighted'))
    print(metrics.accuracy_score(labels_true, newy_labs))




























