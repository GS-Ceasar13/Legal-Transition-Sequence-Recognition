# coding: utf-8

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
import csv
# import gensim
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def train(data_address, data_name,output_dir,latest_checkpoint_file,vector_address=None, embd_dimension=2, embd_dimension_suffix=5, hidden_dim_suffix=2,
          train_splitThreshold=0.8,
          time_unit='day', batch_size=9,
          loss_type='L1Loss', optim_type='Adam', model_type='RNN', hidden_dim=10,    ######################修改model_type名，分别是RNN，LSTM和GRU
          train_type='iteration', n_layer=1, dropout=0.1, max_epoch_num=10, learn_rate_min=0.0001, path_length=8,
          train_record_folder='./train_record/', model_save_folder='./model/', result_save_folder='./result/',
          ts_type='set'):
    # 初始化数据
    out_size = 1
    learn_rate = 0.01
    epoch = 0
    learn_rate_backup = 0.01
    learn_rate_down_backup = 0.001
    loss_deque = deque(maxlen=20)
    loss_change_deque = deque(maxlen=30)
    print('User Model ' + model_type + " To Start Experience.")

    data = InputData(data_address, embd_dimension=embd_dimension)
    print("InputData Finish")

    data.encodeEvent()
    print("EncodeEvent Finish")
    data.encodeTrace()
    print("EncodeTrace Finish")
    # 通过设置固定的随机数种子以获取相同的训练集和测试集,timeEmbedding=data.timeEmbedding
    data.splitData(train_splitThreshold)
    print("SplitData Finish")
    train_singlePrefixData1, train_labelPrefixData, test_singlePrefixData1, test_labelPrefixData, \
               train_singlePrefixData_processing, test_singlePrefixData_processing = data.initBatchData_Prefix()
    print("InitBatchData Finish")
    # data.generateSingleLengthBatch(batch_size)

    # 初始化模型CrossEntropyLoss
    if model_type == 'GRU':
        model=GRU(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer ,dropout=dropout, embedding=data.embedding)
    elif model_type == 'RNN':
        model = RNN(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'LSTM':
        model = LSTM(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim, out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'TRANSFORMER':
        model = nn.Transformer(len(train_singlePrefixData1[0]),nhead=3, dropout=dropout)
    # print("len(train_singlePrefixData1[0]):    $$$$$$$$$", len(train_singlePrefixData1[0]))

    if loss_type == 'L1Loss':
        criterion = nn.L1Loss()
        # criterion = nn.CrossEntropyLoss()
    elif loss_type == 'MSELoss':
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    # 开始训练


    tensor_x = torch.from_numpy(np.array(train_singlePrefixData1).astype(np.float32))
    tensor_y = torch.from_numpy(np.array(train_labelPrefixData).astype(np.float32))
    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    my_dataset_loader = DataLoader(my_dataset, batch_size, drop_last=True, shuffle=False)

    #     训练
    # model = MLP()
    for enpoch in range(100):
        total_loss = torch.FloatTensor([0])
        for i, (input, target) in enumerate(my_dataset_loader):
            optimizer.zero_grad()
            # print("input:              ", np.array(input).shape)

            output = model(Variable(input))
            loss = criterion(output, target)
            # optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.data
            # print(" np.array(input).shape            ", np.array(input).shape)
            # print("input:              ", input)
        if enpoch % 5 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "lr": scheduler.get_last_lr()

            },
            os.path.join(output_dir, latest_checkpoint_file),
        )
        # print("@@@@@@@@@@@@@@@@@@@@@@:   ", total_loss)

def test(data_address, data_name, output_dir, latest_checkpoint_file, vector_address=None, embd_dimension=2,
          embd_dimension_suffix=5, hidden_dim_suffix=2,
          train_splitThreshold=0.8,
          time_unit='day', batch_size=9,
          loss_type='L1Loss', optim_type='Adam', model_type='RNN', hidden_dim=10,
          ######################修改model_type名，分别是RNN，LSTM和GRU
          train_type='iteration', n_layer=1, dropout=0.1, max_epoch_num=10, learn_rate_min=0.0001, path_length=8,
          train_record_folder='./train_record/', model_save_folder='./model/', result_save_folder='./result/',
          ts_type='set'):
    # 初始化数据
    out_size = 1
    learn_rate = 0.01
    epoch = 0
    learn_rate_backup = 0.01
    learn_rate_down_backup = 0.001
    loss_deque = deque(maxlen=20)
    loss_change_deque = deque(maxlen=30)
    print('User Model ' + model_type + " To Start Experience.")

    data = InputData(data_address, embd_dimension=embd_dimension)
    print("InputData Finish")

    data.encodeEvent()
    print("EncodeEvent Finish")
    data.encodeTrace()
    print("EncodeTrace Finish")
    # 通过设置固定的随机数种子以获取相同的训练集和测试集,timeEmbedding=data.timeEmbedding
    data.splitData(train_splitThreshold)
    print("SplitData Finish")
    train_singlePrefixData1, train_labelPrefixData, test_singlePrefixData1, test_labelPrefixData, \
    train_singlePrefixData_processing, test_singlePrefixData_processing = data.initBatchData_Prefix()
    print("InitBatchData Finish")
    # data.generateSingleLengthBatch(batch_size)

    # 初始化模型CrossEntropyLoss
    if model_type == 'GRU':
        model = GRU(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim,
                    out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'RNN':
        model = RNN(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim,
                    out_size=out_size,
                    batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'LSTM':
        model = LSTM(len(train_singlePrefixData1[0]), embedding_dim=embd_dimension, hidden_dim=hidden_dim,
                     out_size=out_size,
                     batch_size=batch_size, n_layer=n_layer, dropout=dropout, embedding=data.embedding)
    elif model_type == 'TRANSFORMER':
        model = nn.Transformer(len(train_singlePrefixData1[0]), nhead=3,dropout=dropout)
    # print("len(train_singlePrefixData1[0]):    $$$$$$$$$", len(train_singlePrefixData1[0]))

    if loss_type == 'L1Loss':
        criterion = nn.L1Loss()
        # criterion = nn.CrossEntropyLoss()
    elif loss_type == 'MSELoss':
        criterion = nn.MSELoss()

    ckpt = torch.load(os.path.join(output_dir, latest_checkpoint_file),
                      map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])
    # 开始训练

    tensor_x = torch.from_numpy(np.array(train_singlePrefixData1).astype(np.float32))
    tensor_y = torch.from_numpy(np.array(train_labelPrefixData).astype(np.float32))
    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    my_dataset_loader = DataLoader(my_dataset, batch_size, drop_last=True, shuffle=False)

    #     训练
    # model = MLP()
    ACC=0
    F1_EVERY_CLASS=0
    F1_MACRO=0
    F1_MICRO=0
    PREC_EVERY_CLASS = 0
    PREC_MACRO = 0
    PREC_MICRO = 0
    RECALL_EVERY_CLASS = 0
    RECALL_MACRO = 0
    RECALL_MICRO = 0
    ROC_AUC_EVERY_CLASS = 0
    ROC_AUC_MACRO = 0
    ROC_AUC_MICRO = 0
    count=0
    for i, (input, target) in enumerate(my_dataset_loader):
        # print("input:              ", np.array(input).shape)
        output = model(input)
        output1=output.detach().numpy()
        for line in range(len(output1)):
            if output1[line][0]<0.5:
                output1[line][0]=0
            if output1[line][0]>=0.5:
                output1[line][0]=1
        acc = accuracy_score(target.detach().numpy(), output1)
        f1_every_class = f1_score(target.detach().numpy(), output1,
                                  average=None, zero_division=1)
        f1_macro = f1_score(target.detach().numpy(), output1,
                            average='macro', zero_division=1)
        f1_micro = f1_score(target.detach().numpy(), output1,
                            average='micro', zero_division=1)
        prec_micro= precision_score(target.detach().numpy(), output1,average='micro',zero_division=1)
        prec_macro = precision_score(target.detach().numpy(), output1, average='macro', zero_division=1)
        prec_every_class = precision_score(target.detach().numpy(), output1, average=None, zero_division=1)

        recall_micro=recall_score(target.detach().numpy(), output1,average='micro',zero_division=1)
        recall_macro = recall_score(target.detach().numpy(), output1, average='macro', zero_division=1)
        recall_every_class = recall_score(target.detach().numpy(), output1, average=None, zero_division=1)

        # roc_auc_micro=roc_auc_score(target.detach().numpy(), output1,average='micro')
        # roc_auc_macro = roc_auc_score(target.detach().numpy(), output1, average='macro')
        # roc_auc_every_class = roc_auc_score(target.detach().numpy(), output1, average=None)
        # print(acc)
        ACC=ACC+acc
        F1_EVERY_CLASS = F1_EVERY_CLASS+f1_every_class
        F1_MACRO = F1_MACRO+f1_macro
        F1_MICRO = F1_MICRO+f1_micro
        PREC_EVERY_CLASS = PREC_EVERY_CLASS+prec_every_class
        PREC_MACRO = PREC_MACRO+prec_macro
        PREC_MICRO = PREC_MICRO+prec_micro
        RECALL_EVERY_CLASS = RECALL_EVERY_CLASS+recall_every_class
        RECALL_MACRO = RECALL_MACRO+recall_macro
        RECALL_MICRO = RECALL_MICRO+recall_micro
        # ROC_AUC_EVERY_CLASS = ROC_AUC_EVERY_CLASS+roc_auc_every_class
        # ROC_AUC_MACRO = ROC_AUC_MACRO+roc_auc_macro
        # ROC_AUC_MICRO = ROC_AUC_MICRO+roc_auc_micro
        count+=1
    # f1=open('./model/gru_metric_result10_1%.csv','a+')
    # print("ACC:",ACC/count,file=f1)
    # print("F1_EVERY_CLASS:",F1_EVERY_CLASS/count,file=f1)
    # print("F1_MACRO:", F1_MACRO/ count, file=f1)
    # print("F1_MICRO:", F1_MICRO / count, file=f1)
    # print("PREC_EVERY_CLASS:", PREC_EVERY_CLASS / count, file=f1)
    # print("PREC_MACRO:", PREC_MACRO / count, file=f1)
    # print("PREC_MICRO:", PREC_MICRO / count, file=f1)
    # print("RECALL_EVERY_CLASS:", RECALL_EVERY_CLASS / count, file=f1)
    # print("RECALL_MACRO:", RECALL_MACRO / count, file=f1)
    # print("RECALL_MICRO:", RECALL_MICRO / count, file=f1)
    # f1.close()
    with open('./result/metric_result_0%.csv', 'a+', newline='') as f:
        f_csv = csv.writer(f)
        row = []
        row.append(ACC/count)
        row.append(F1_EVERY_CLASS/count)
        row.append(F1_MACRO/ count)
        row.append(F1_MICRO / count)
        row.append(PREC_EVERY_CLASS / count)
        row.append(PREC_MACRO / count)
        row.append(PREC_MICRO / count)
        row.append(RECALL_EVERY_CLASS / count)
        row.append(RECALL_MACRO / count)
        row.append(RECALL_MICRO / count)
        f_csv.writerow(row)
        row.clear()
            # print(" np.array(input).shape            ", np.array(input).shape)
            # print("input:              ", input)



    # trainGRNN = []
    # trainGRNN_label = []
    # testGRNN = []
    # testGRNN_label = []
    #
    # for i, (x_, y_) in enumerate(my_dataset_loader):
    #     # x_ = np.array(x_)
    #     # x_ = Variable(torch.LongTensor(x_))
    #     output = model(Variable(x_))
    #     trainGRNN.append(output.detach().numpy().tolist())
    #
    # print("trainGRNN:::::              ", np.array(trainGRNN).shape)
    # print("type(trainGRNN):::::              ", type(trainGRNN))
    # tensor_x = torch.from_numpy(np.array(test_singlePrefixData1).astype(np.float32))
    # tensor_y = torch.from_numpy(np.array(test_labelPrefixData).astype(np.float32))
    # my_test_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    # my_test_dataset_loader = DataLoader(my_test_dataset, batch_size, drop_last=True, shuffle=False)
    #
    # for i, (x_, y_) in enumerate(my_test_dataset_loader):
    #     # x_ = np.array(x_)
    #     # x_ = Variable(torch.LongTensor(x_))
    #     output = model(Variable(x_))
    #     testGRNN.append(output.detach().numpy().tolist())
    #
    #
    # f = open("/Users/caorui/Desktop/code5-NextActivity_NonPartition/score/" + model_type + "_final" + "_"+ data_name + ".txt",
    #     "a")
    #
    #
    # trainGRNNarray = np.array(trainGRNN).reshape((-1, hidden_dim))
    # # trainGRNNarray = np.array(trainGRNN).reshape((len(trainGRNN),-1))
    # trainGRNN_labelarray = np.array(train_singlePrefixData_processing).reshape(len(train_singlePrefixData_processing), 1)
    # testGRNNarray = np.array(testGRNN).reshape((-1, hidden_dim))
    # # testGRNNarray = np.array(testGRNN).reshape((len(testGRNN), -1))
    # testGRNN_labelarray = np.array(test_singlePrefixData_processing).reshape(len(test_singlePrefixData_processing), 1)
    # print(len(train_singlePrefixData1))
    # print("trainGRNNarray:::::              ", np.array(trainGRNNarray).shape)
    # print("type(trainGRNNarray):::::              ", type(trainGRNNarray))
    # print("trainGRNN_labelarray:::::              ", np.array(trainGRNN_labelarray).shape)
    # print("type(trainGRNN_labelarray):::::              ", type(trainGRNN_labelarray))
    # print(0)
    #
    #
    #
    # data_train=trainGRNNarray
    # data_test = testGRNNarray
    # y=trainGRNN_labelarray
    # y1=testGRNN_labelarray
    #
    #
    # y = y.ravel()
    # y1=y1.ravel()
    # # print("y1:::::::::::;   ", y1)
    #
    #
    #
    # for num in range(len(y) % 9):
    #     y = np.delete(y, -1, 0)
    # for num in range(len(y1) % 9):
    #     y1 = np.delete(y1, -1, 0)
    # print("MLP Data OK")################################################################################################
    #
    #
    #
    # #    不同的分类器比较， 可以作为对比实验之一。说明我们选择的分类器具有最好的分类效果。
    # # model = MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate="adaptive") #   1。
    # # model = MLPRegressor(hidden_layer_sizes=(100, 100), learning_rate="adaptive")  # 2。 # MLPRegressor比MLPClassifier的效果差，所以不用作分类器。
    # # model = LogisticRegression()  #   线性分类                                        3。
    # # model = SVR(kernel='rbf')  #     rbf SVR     高斯RBF核函数(高斯径向基函数)            4。
    # # model = SVR(kernel='linear')  #     linear SVRp    线性核函数                      5。
    # # model = SVR(kernel='poly')  #      poly 多项式核函数                               6。
    # model = SVR(kernel='sigmoid')  #     sigmoid核函数                                7。
    #
    #
    #
    # model.fit(data_train, y)
    #
    # print("MLP test...")
    # # model.eval()  # 将模型改为预测模式
    # y_predict = model.predict(data_test)
    # # y_predict = np.around(y_predict)
    # y_predict = y_predict.astype(np.int64)
    #
    # print("accuracy_score is", accuracy_score(y1, y_predict))
    # print("precision_score is", precision_score(y1, y_predict, average='macro'))
    # print("recall_score is", recall_score(y1, y_predict, average='macro'))
    # print("f1_score is", f1_score(y1, y_predict, average='macro'))
    # listscore = []
    # listscore.append(accuracy_score(y1, y_predict))
    # listscore.append(precision_score(y1, y_predict, average='macro'))
    # listscore.append(recall_score(y1, y_predict, average='macro'))
    # listscore.append(f1_score(y1, y_predict, average='macro'))
    # f.write(str(listscore) + '\n')
    # f.close()




if __name__ == '__main__':
    # #       BPIC2017A.csv   BPIC2017O1.csv    BPIC2017W.csv
#     BPIC_2019_A11.csv   BPIC_2019_O.csv   BPIC_2019_W1.csv
#     train(data_address='E:/PycharmProjects/code5-Classifer/Data/result10_0%.csv',  # 修改数据名
#           data_name='result5_1%', embd_dimension=2, ################################################## 修改数据名
#           train_splitThreshold=0.8, #这是为了把所有数据都训练模型，目的是保存隐状态。
#           batch_size=9, n_layer=1, hidden_dim=1,
#           loss_type='L1Loss', optim_type='Adam', model_type='RNN', ######################修改model_type名，分别是RNN，LSTM和GRU
#           train_type='iteration',output_dir='./model',latest_checkpoint_file='rnnlatest_result10_0%.pt')
    test(data_address='E:/PycharmProjects/code5-Classifer/Data/result2_30%.csv',  # 修改数据名
          data_name='result10', embd_dimension=2,  ################################################## 修改数据名
          train_splitThreshold=0.8,  # 这是为了把所有数据都训练模型，目的是保存隐状态。
          batch_size=9, n_layer=1, hidden_dim=1,
          loss_type='L1Loss', optim_type='Adam', model_type='GRU',  ######################修改model_type名，分别是RNN，LSTM和GRU
          train_type='iteration', output_dir='./model', latest_checkpoint_file='grulatest_result2_30%.pt')