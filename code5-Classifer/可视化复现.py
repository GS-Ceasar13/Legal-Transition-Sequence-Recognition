import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":

    label_train = torch.load("label_train_LSTM_bigger-exampleConCycle.pth")
    y = np.array(label_train).reshape((len(label_train), 1))

    color = ['r', 'g', 'b', 'c', 'y', 'm', 'k', '#ff00ff', '#99ff66','#8C7853']
    shape = ['o', '^', '+', '<', '>', 'x', 's', 'd', 'p', 'h']
    color_test = []


    finalData =torch.load("myfinalHideState2wei_LSTM_bigger-exampleConCycle.pth")
    for i in range(len(finalData)):
        color_test.append(color[int(y[i])])
    finalData1=np.array(finalData[:, 0])
    finalData2 = np.array(finalData[:, 1])
    # plt.scatter(finalData1, finalData2, marker='o', color=color_test[:])
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析o.eps', format='eps', dpi=1000)
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析o.pdf', format='pdf', dpi=1000)
    # plt.scatter(finalData1, finalData2, marker='^', color=color_test[:])
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析^.eps', format='eps', dpi=1000)
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析^.pdf', format='pdf', dpi=1000)
    # plt.scatter(finalData1, finalData2, marker='+', color=color_test[:])
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析+.eps', format='eps', dpi=1000)
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析+.pdf', format='pdf', dpi=1000)
    # plt.scatter(finalData1, finalData2, marker='*', color=color_test[:])
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析星号.eps', format='eps', dpi=1000)
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析星号.pdf', format='pdf', dpi=1000)
    # plt.scatter(finalData1, finalData2, marker='.', color=color_test[:])
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析点.eps', format='eps', dpi=1000)
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析点.pdf', format='pdf', dpi=1000)
    # plt.scatter(finalData1, finalData2, marker='x', color=color_test[:])
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析x.eps', format='eps', dpi=1000)
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析x.pdf', format='pdf', dpi=1000)
    # plt.scatter(finalData1, finalData2, marker='s', color=color_test[:])
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析s.eps', format='eps', dpi=1000)
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析s.pdf', format='pdf', dpi=1000)
    # plt.scatter(finalData1, finalData2, marker='d', color=color_test[:])
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析d.eps', format='eps', dpi=1000)
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析d.pdf', format='pdf', dpi=1000)
    # plt.scatter(finalData1, finalData2, marker='p', color=color_test[:])
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析p.eps', format='eps', dpi=1000)
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析p.pdf', format='pdf', dpi=1000)
    # plt.scatter(finalData1, finalData2, marker='h', color=color_test[:])
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析h.eps', format='eps', dpi=1000)
    # plt.savefig('./img/主成分分析/2017A8GRU主成分分析h.pdf', format='pdf', dpi=1000)

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
            type0_x.append(finalData[i,0])
            type0_y.append(finalData[i,1])
        if y[i] == 1:  # 第i行的label为2时
            type1_x.append(finalData[i,0])
            type1_y.append(finalData[i,1])
        if y[i] == 2:  # 第i行的label为3时
            type2_x.append(finalData[i,0])
            type2_y.append(finalData[i,1])
        if y[i] == 3:  # 第i行的label为1时
            type3_x.append(finalData[i,0])
            type3_y.append(finalData[i,1])
        if y[i] == 4:  # 第i行的label为2时
            type4_x.append(finalData[i,0])
            type4_y.append(finalData[i,1])
        if y[i] == 5:  # 第i行的label为3时
            type5_x.append(finalData[i,0])
            type5_y.append(finalData[i,1])
        if y[i] == 6:  # 第i行的label为3时
            type6_x.append(finalData[i,0])
            type6_y.append(finalData[i,1])
        if y[i] == 7:  # 第i行的label为3时
            type7_x.append(finalData[i,0])
            type7_y.append(finalData[i,1])
    type0 = plt.scatter(type0_x, type0_y, marker=shape[0], s=30, c=color[0])
    type1 = plt.scatter(type1_x, type1_y, marker=shape[1], s=30, c=color[1])
    type2 = plt.scatter(type2_x, type2_y, marker=shape[2], s=30, c=color[2])
    type3 = plt.scatter(type3_x, type3_y, marker=shape[3], s=30, c=color[3])
    type4 = plt.scatter(type4_x, type4_y, marker=shape[4], s=30, c=color[4])
    type5 = plt.scatter(type5_x, type5_y, marker=shape[5], s=30, c=color[5])
    type6 = plt.scatter(type6_x, type6_y, marker=shape[6], s=30, c=color[6])
    type7 = plt.scatter(type7_x, type7_y, marker=shape[7], s=30, c=color[7])
    plt.savefig('./img/主成分分析/PCA_LSTM_bigger-exampleConCycle.eps', format='eps', dpi=1000)
    plt.savefig('./img/主成分分析/PCA_LSTM_bigger-exampleConCycle.pdf', format='pdf', dpi=1000)
    plt.savefig('./img/主成分分析/PCA_LSTM_bigger-exampleConCycle.png', format='png', dpi=1000)
    plt.savefig('./img/主成分分析/PCA_LSTM_bigger-exampleConCycle.svg', format='svg', dpi=1000)







