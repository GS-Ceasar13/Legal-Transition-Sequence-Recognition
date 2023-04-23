import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import torch
import torch.nn as nn
#删除前9行
s = torch.load("E:/PycharmProjects/code5-Classifer/label_test_GRU_bigger-exampleConCycle.pth")
# for i in range(14):
#     print(s[i])
# for i in range(9):
#     s=np.delete(s,0,0)
# print(len(s))
# print(s.shape)
# print(type(s))
# for i in range(5):
#     print(s[i])
print(s)
# torch.save(s,"myfinalHideState5weiLSTMhelpdesk1.pth")
#
# #读取类别
# s = torch.load("label_train2019A118.pth")
# set1=set()
# for i in s:
#     if i not in set1:
#         set1.add(i)
# print(len(set1))
# print(set1)

#画图
# data_train= torch.load("myfinalHideState2weit-sne2017A8.pth")
#
# print(data_train)
# print(len(data_train))
# label_train= torch.load("label_trainLSTM2017A8.pth")
# y=np.array(label_train).reshape((len(label_train),1))
# print(len(y))
# for i in range(len(data_train)-len(y)):
#     data_train = np.delete(data_train, -1, axis=0)
# print(len(data_train))
# color = ['r', 'g', 'b', 'c', 'y', 'm', 'k','#ff00ff','#99ff66']
# shape=['o','^','+','*','.','x','s','d','p','h']
# color_test=[]
# for i in range(len(data_train)):
#    color_test.append(color[int(y[i])])
#
# plt.scatter(data_train[:, 0], data_train[:, 1], marker='o',color=color_test[:])
# plt.show()

# #保留4位
# s = torch.load("myfinalHideState5weiLSTM2017O1.pth")
# print(s)
# print(type(s))
# for i in range(len(s)):
#     s[i][0]=format(s[i][0], '.3f')
#     s[i][1] = format(s[i][1], '.3f')
#     s[i][2] = format(s[i][2], '.3f')
#     s[i][3] = format(s[i][3], '.3f')
#     s[i][4] = format(s[i][4], '.3f')
# print(s)
# torch.save(s,"myfinalHideState5weiLSTM2017O1.pth")

# print(w)
# print(len(w))
# q = torch.load("label_test.pth")
# w = torch.load("label_train.pth")

# q = s.detach().numpy().reshape(9, 5)
# batch=[]
# for line in q:
#     print(line)
#     t=[]
#     x=line[0]
#     y=line[2]
#     z=line[4]
#     t.append(x)
#     t.append(y)
#     t.append(z)
#     batch.append(t)
# torch.save(batch, "myfinalHideState.pth")

# print(batch)
# train_label = torch.load("label_train.pth")
# test_label = torch.load("label_test.pth")
# train_label.extend(test_label)
# torch.save(train_label, "label.pth")
# e = torch.load("label.pth")
# print(e)
# print(type(e))
# print(q)
# print(len(train_label))
# print(type(train_label))
# print(w)
# print(len(test_label))
# print(type(test_label))

# print(type(e))
# print(type(batch))
# print(s)
# print(type(q))