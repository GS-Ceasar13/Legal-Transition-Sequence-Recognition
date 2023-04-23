
import numpy as np
import os
import torch
import torch.nn as nn

import torch.optim as optim
from torch.autograd import Variable
# -*- coding: utf-8 -*-
# import IOUtil as iou
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from decimal import Decimal
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

num = []

#### 训练%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                修改3个地方

print("MLP start")


data_train= torch.load("myscore5weiGRU_Loan_application_Configuration1.pth")
# for i in range(9):
#     data_train.pop(0)
label_train=torch.load("label_trainGRU_Loan_application_Configuration1.pth")
f=open("F:/代码/context-sensitive-deep-learning-comparison/师姐任务/score/GRU_Loan_application_Configuration1_score.txt","a")
print(len(label_train))
print(label_train)
y=np.array(label_train).reshape((len(label_train),1))

for num in range(len(y)%9):
    y=np.delete(y,-1,0)

for i in range(len(data_train)-len(y)):
    data_train = np.delete(data_train, -1, axis=0)

temp=data_train


# x, y = np.split(data, (4,), axis=1)
print(len(y))
x = temp
print(x)
print(type(x))


# X = iou.readArray('X_train.csv')
# y = iou.readArray('y_train.csv')
y = y.ravel()
print("MLP Data OK")



model = MLPClassifier(hidden_layer_sizes=(100,100), learning_rate="adaptive")
model.fit(x, y)

######预测

# X_test = iou.readArray('X_test.csv')
# y_test = iou.readArray('y_test.csv')

print("MLP predict...")

y_predict = model.predict(x)
y_predict=np.around(y_predict)
y_predict=y_predict.astype(np.int64)
print(y_predict)
print(len(y_predict))
print(type(y_predict))


# print('model.score(x,y):      ', model.accscore(x,y))
print("accuracy_score is",accuracy_score(y,y_predict))
print("precision_score is",precision_score(y,y_predict,average='macro'))
print("recall_score is",recall_score(y,y_predict,average='macro'))
print("f1_score is",f1_score(y,y_predict,average='macro'))
listscore=[]
listscore.append(accuracy_score(y,y_predict))
listscore.append(precision_score(y,y_predict,average='macro'))
listscore.append(recall_score(y,y_predict,average='macro'))
listscore.append(f1_score(y,y_predict,average='macro'))
f.write(str(listscore)+'\n')
f.close()



# color = ['r', 'g', 'b', 'c', 'y', 'm', 'k','#ff00ff','#99ff66','#8C7853']
# shape=['o','^','+','*','.','x','s','d','p','h']
# color_test=[]
# for i in range(len(x)):
#    color_test.append(color[y_predict[i]])
# print(color_test)
# shape_test=[]
# for i in range(len(x_test)):
#    shape_test.append(shape[y_hat2[i]])
# print(shape_test)
#
# alpha = 0.5
# for line in range(len(x_test)):
#     plt.scatter(x_test[line, 0], x_test[line, 1], marker=shape[y_hat2[line]],color=color[y_hat2[line]])
##############################################################################################################################
print('end')
print("MLP Done!!!!!!")

