from LSTM import LSTM
from GRU import GRU
from RNN import RNN
# from MLP_NP import MLP
# from model_final import MLP
from input_final import InputData
from collections import deque
import numpy as np
import os
import torch
import torch.nn as nn
# import gensim
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


SEED = 13
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

model = MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate="adaptive")  # 1。
# model = MLPRegressor(hidden_layer_sizes=(100, 100), learning_rate="adaptive")  # 2。 # MLPRegressor比MLPClassifier的效果差，所以不用作分类器。
# model = LogisticRegression()  #   线性分类                                        3。
# model = SVR(kernel='rbf')  #     rbf SVR     高斯RBF核函数(高斯径向基函数)            4。
# model = SVR(kernel='linear')  #     linear SVRp    线性核函数                      5。
# model = SVR(kernel='poly')  #      poly 多项式核函数                               6。
# model = SVR(kernel='sigmoid')  #     sigmoid核函数                                7。

data_train = torch.load("myfinalHideState2wei_RNN_repairExample_cycle.pth")
lab= torch.load("label_train_RNN_repairExample_cycle.pth")
model.fit(data_train, lab)

print(0)

print("MLP test...")
# model.eval()  # 将模型改为预测模式
y_predict = model.predict(data_train)
# y_predict = np.around(y_predict)
y_predict = y_predict.astype(np.int64)

y = np.array(y_predict).reshape((len(y_predict), 1))
torch.save(y, "mlppredict_RNN_repairExample_cycle.pth")
color = ['r', 'g', 'b', 'c', 'y', 'm', 'k', '#ff00ff', '#99ff66','#8C7853']
shape = ['o', '^', '+', '<', '>', 'x', 's', 'd', 'p', 'h']
color_test = []
for i in range(len(data_train)):
        color_test.append(color[int(y[i])])

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
        type0_x.append(data_train[i, 0])
        type0_y.append(data_train[i, 1])
    if y[i] == 1:  # 第i行的label为2时
        type1_x.append(data_train[i, 0])
        type1_y.append(data_train[i, 1])
    if y[i] == 2:  # 第i行的label为3时
        type2_x.append(data_train[i, 0])
        type2_y.append(data_train[i, 1])
    if y[i] == 3:  # 第i行的label为4时
        type3_x.append(data_train[i, 0])
        type3_y.append(data_train[i, 1])
    if y[i] == 4:  # 第i行的label为5时
        type4_x.append(data_train[i, 0])
        type4_y.append(data_train[i, 1])
    if y[i] == 5:  # 第i行的label为6时
        type5_x.append(data_train[i, 0])
        type5_y.append(data_train[i, 1])
    if y[i] == 6:  # 第i行的label为7时
        type6_x.append(data_train[i, 0])
        type6_y.append(data_train[i, 1])
    if y[i] == 7:  # 第i行的label为8时
        type7_x.append(data_train[i, 0])
        type7_y.append(data_train[i, 1])
type0 = plt.scatter(type0_x, type0_y, marker=shape[0], s=30, c=color[0])
type1 = plt.scatter(type1_x, type1_y, marker=shape[1], s=30, c=color[1])
type2 = plt.scatter(type2_x, type2_y, marker=shape[2], s=30, c=color[2])
type3 = plt.scatter(type3_x, type3_y, marker=shape[3], s=30, c=color[3])
type4 = plt.scatter(type4_x, type4_y, marker=shape[4], s=30, c=color[4])
type5 = plt.scatter(type5_x, type5_y, marker=shape[5], s=30, c=color[5])
type6 = plt.scatter(type6_x, type6_y, marker=shape[6], s=30, c=color[6])
type7 = plt.scatter(type7_x, type7_y, marker=shape[7], s=30, c=color[7])
plt.savefig('./img/主成分分析/MLP_RNN_repairExample_cycle.eps', format='eps', dpi=1000)
plt.savefig('./img/主成分分析/MLP_RNN_repairExample_cycle.pdf', format='pdf', dpi=1000)
plt.savefig('./img/主成分分析/MLP_RNN_repairExample_cycle.png', format='png', dpi=1000)
plt.savefig('./img/主成分分析/MLP_RNN_repairExample_cycle.svg', format='svg', dpi=1000)
