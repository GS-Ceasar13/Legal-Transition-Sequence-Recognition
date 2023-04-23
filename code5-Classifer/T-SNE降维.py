from sklearn.manifold import TSNE
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time

datafile =torch.load("/Users/caorui/Desktop/code5-NextActivity_NonPartition/myfinalHideState5weiLSTMrepairExampleConcurrency.pth")
label_train=torch.load("/Users/caorui/Desktop/code5-NextActivity_NonPartition/label_trainLSTMrepairExampleConcurrency.pth")
y=np.array(label_train).reshape((len(label_train),1))


# data_l = DataFrame(datafile)
print("data_l ok")
dataMat = np.array(datafile)
for i in range(len(dataMat)-len(label_train)):
    dataMat = np.delete(dataMat, -1, axis=0)
print(len(dataMat))
# print(dataMat)
pca_tsne = TSNE(n_components=2)
newMat = pca_tsne.fit_transform(dataMat)

# data1 = DataFrame(newMat)
torch.save(newMat,"myfinalHideState2weiLSTMt-snerepairExampleConcurrency.pth")

color = ['r', 'g', 'b', 'c', 'y', 'm', 'k','#ff00ff','#99ff66','#8C7853']
shape=['o','^','+','*','.','x','s','d','p','h']
color_test=[]
for i in range(len(dataMat)):
   color_test.append(color[int(y[i])])

plt.scatter(newMat[:, 0], newMat[:, 1], marker='o',color=color_test[:])
# for i in range(len(dataMat)):
#     plt.annotate("%s" % y[i], xy=(newMat[i, 0], newMat[i, 1]), xytext=(-20, 10), textcoords='offset points')

# plt.show()
plt.savefig("LSTMt-snerepairExampleConcurrency.png")

# data1.to_csv('2.csv',index=False,header=False)