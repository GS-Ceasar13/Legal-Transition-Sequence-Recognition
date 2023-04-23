# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:04:26 2016
PCA source code
@author: liudiwei
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# 计算均值,要求输入数据为numpy的矩阵格式，行表示样本数，列表示特征
def meanX(dataX):
    return np.mean(dataX, axis=0)  # axis=0表示按照列来求均值，如果输入list,则axis=1


# 计算方差,传入的是一个numpy的矩阵格式，行表示样本数，列表示特征
def variance(X):
    m, n = np.shape(X)
    mu = meanX(X)
    muAll = np.tile(mu, (m, 1))
    X1 = X - muAll
    variance = 1. / m * np.diag(X1.T * X1)
    return variance


# 标准化,传入的是一个numpy的矩阵格式，行表示样本数，列表示特征
def normalize(X):
    m, n = np.shape(X)
    mu = meanX(X)
    muAll = np.tile(mu, (m, 1))
    X1 = X - muAll
    X2 = np.tile(np.diag(X.T * X), (m, 1))
    XNorm = X1 / X2
    return XNorm


"""
参数：
	- XMat：传入的是一个numpy的矩阵格式，行表示样本数，列表示特征    
	- k：表示取前k个特征值对应的特征向量
返回值：
	- finalData：参数一指的是返回的低维矩阵，对应于输入参数二
	- reconData：参数二对应的是移动坐标轴后的矩阵
"""


def pca(XMat, k):
    average = meanX(XMat)
    m, n = np.shape(XMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)  # 计算协方差矩阵
    featValue, featVec = np.linalg.eig(covX)  # 求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue)  # 按照featValue进行从大到小排序
    finalData = []
    if k > n:
        print("k must lower than feature number")
        return
    else:
        # 注意特征向量时列向量，而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.matrix(featVec.T[index[:k]])  # 所以这里需要进行转置
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return finalData, reconData


def loaddata(datafile):
    dataarray=torch.load(datafile)
    return np.array(dataarray).astype(np.float)


def plotBestFit(data1, data2):
    dataArr1 = np.array(data1)
    dataArr2 = np.array(data2)

    m = np.shape(dataArr1)[0]
    axis_x1 = []
    axis_y1 = []
    axis_x2 = []
    axis_y2 = []
    for i in range(m):
        axis_x1.append(dataArr1[i, 0])
        axis_y1.append(dataArr1[i, 1])
        axis_x2.append(dataArr2[i, 0])
        axis_y2.append(dataArr2[i, 1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x1, axis_y1, s=50, c='red', marker='s')
    ax.scatter(axis_x2, axis_y2, s=50, c='blue')
    plt.xlabel('x1');
    plt.ylabel('x2');
    plt.savefig("outfile.png")
    plt.show()


# # 简单测试
# # 数据来源：http://www.cnblogs.com/jerrylead/archive/2011/04/18/2020209.html
# def test():
#     X = [[2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1],
#          [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]]
#     XMat = np.matrix(X).T
#     k = 2
#     return pca(XMat, k)


# 根据数据集data.txt
# def main():
#     datafile = "myfinalHideState5wei.pth"
#     XMat = loaddata(datafile)
#     k = 2
#     return pca(XMat, k)


if __name__ == "__main__":
    datafile = "myfinalHideState5weiRNNhelpdesk1.pth"
    XMat = loaddata(datafile)
    k = 2

    # plotBestFit(finalData, reconMat)

    label_train = torch.load("label_train_RNN_helpdesk1.pth")
    y = np.array(label_train).reshape((len(label_train), 1))

    # data_l = DataFrame(datafile)
    dataMat = np.array(XMat)
    for i in range(len(dataMat) - len(label_train)):
        dataMat = np.delete(dataMat, -1, axis=0)
    print(len(dataMat))

    color = ['r', 'g', 'b', 'c', 'y', 'm', 'k', '#ff00ff', '#99ff66','#8C7853']
    shape = ['o', '^', '+', '<', '>', 'x', 's', 'd', 'p', 'h']
    color_test = []
    for i in range(len(dataMat)):
        color_test.append(color[int(y[i])])

    finalData, reconMat = pca(dataMat, k)
    print(finalData)
    print(type(finalData))
    torch.save(finalData, "myfinalHideState2wei_RNN_helpdesk1.pth")
    # finalData1=np.array(finalData[:, 0])
    # finalData2 = np.array(finalData[:, 1])
    # plt.scatter(finalData1, finalData2, marker='o', color=color_test[:])

    # for i in range(len(dataMat)):
    #     plt.annotate("%s" % y[i], xy=(finalData[i, 0], finalData[i, 1]), xytext=(-20, 10), textcoords='offset points')
    # type0_x = []
    # type0_y = []
    # type1_x = []
    # type1_y = []
    # type2_x = []
    # type2_y = []
    # type3_x = []
    # type3_y = []
    # type4_x = []
    # type4_y = []
    # type5_x = []
    # type5_y = []
    # type6_x = []
    # type6_y = []
    # type7_x = []
    # type7_y = []
    #
    #
    # for i in range(len(y)):
    #     if y[i] == 0:  # 第i行的label为1时
    #         type0_x.append(finalData[i,0])
    #         type0_y.append(finalData[i,1])
    #     if y[i] == 1:  # 第i行的label为2时
    #         type1_x.append(finalData[i,0])
    #         type1_y.append(finalData[i,1])
    #     if y[i] == 2:  # 第i行的label为3时
    #         type2_x.append(finalData[i,0])
    #         type2_y.append(finalData[i,1])
    #     if y[i] == 3:  # 第i行的label为1时
    #         type3_x.append(finalData[i,0])
    #         type3_y.append(finalData[i,1])
    #     if y[i] == 4:  # 第i行的label为2时
    #         type4_x.append(finalData[i,0])
    #         type4_y.append(finalData[i,1])
    #     if y[i] == 5:  # 第i行的label为3时
    #         type5_x.append(finalData[i,0])
    #         type5_y.append(finalData[i,1])
    #     if y[i] == 6:  # 第i行的label为3时
    #         type6_x.append(finalData[i,0])
    #         type6_y.append(finalData[i,1])
    #     if y[i] == 7:  # 第i行的label为3时
    #         type7_x.append(finalData[i,0])
    #         type7_y.append(finalData[i,1])
    #
    #
    # # fig = plt.figure(figsize=(10, 6))
    # # ax = fig.add_subplot(111)
    #
    # type0 = plt.scatter(type0_x, type0_y, s=30, c=color[0])
    # type1 = plt.scatter(type1_x, type1_y, s=30, c=color[1])
    # type2 = plt.scatter(type2_x, type2_y, s=30, c=color[2])
    # type3 = plt.scatter(type3_x, type3_y, s=30, c=color[3])
    # type4 = plt.scatter(type4_x, type4_y, s=30, c=color[4])
    # type5 = plt.scatter(type5_x, type5_y, s=30, c=color[5])
    # type6 = plt.scatter(type6_x, type6_y, s=30, c=color[6])
    # type7 = plt.scatter(type7_x, type7_y, s=30, c=color[7])
    #
    #
    #
    # plt.legend((type1, type2, type3, type4, type5, type6), ("1", "2", "3", "4", "5", "6"), loc=0)


    # plt.show()
    # plt.savefig("LSTMzhuchengfenrepairExampleConcurrency.png")
    type0_x = []
    type0_y = []
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    type5_x = []
    type5_y = []
    type6_x = []
    type6_y = []
    type7_x = []
    type7_y = []

    for i in range(len(y)):
        if y[i] == 0:  # 第i行的label为1时
            type0_x.append(finalData[i, 0])
            type0_y.append(finalData[i, 1])
        if y[i] == 1:  # 第i行的label为2时
            type1_x.append(finalData[i, 0])
            type1_y.append(finalData[i, 1])
        if y[i] == 2:  # 第i行的label为3时
            type2_x.append(finalData[i, 0])
            type2_y.append(finalData[i, 1])
        if y[i] == 3:  # 第i行的label为4时
            type3_x.append(finalData[i, 0])
            type3_y.append(finalData[i, 1])
        if y[i] == 4:  # 第i行的label为5时
            type4_x.append(finalData[i, 0])
            type4_y.append(finalData[i, 1])
        if y[i] == 5:  # 第i行的label为6时
            type5_x.append(finalData[i, 0])
            type5_y.append(finalData[i, 1])
        if y[i] == 6:  # 第i行的label为7时
            type6_x.append(finalData[i, 0])
            type6_y.append(finalData[i, 1])
        if y[i] == 7:  # 第i行的label为8时
            type7_x.append(finalData[i, 0])
            type7_y.append(finalData[i, 1])
    type0 = plt.scatter(type0_x, type0_y, marker=shape[0], s=30, c=color[0])
    type1 = plt.scatter(type1_x, type1_y, marker=shape[1], s=30, c=color[1])
    type2 = plt.scatter(type2_x, type2_y, marker=shape[2], s=30, c=color[2])
    type3 = plt.scatter(type3_x, type3_y, marker=shape[3], s=30, c=color[3])
    type4 = plt.scatter(type4_x, type4_y, marker=shape[4], s=30, c=color[4])
    type5 = plt.scatter(type5_x, type5_y, marker=shape[5], s=30, c=color[5])
    type6 = plt.scatter(type6_x, type6_y, marker=shape[6], s=30, c=color[6])
    type7 = plt.scatter(type7_x, type7_y, marker=shape[7], s=30, c=color[7])
    plt.savefig('./img/主成分分析/PCA_RNN_helpdesk1.eps', format='eps', dpi=1000)
    plt.savefig('./img/主成分分析/PCA_RNN_helpdesk1.pdf', format='pdf', dpi=1000)
    plt.savefig('./img/主成分分析/PCA_RNN_helpdesk1.png', format='png', dpi=1000)
    plt.savefig('./img/主成分分析/PCA_RNN_helpdesk1.svg', format='svg', dpi=1000)