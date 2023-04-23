# coding: utf-8
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from math import sqrt
import random


random.seed = 13


class InputData():   # 前缀的处理，也可以处理后缀
    def __init__(self, data_address, embd_dimension=2):
        self.embedding = None
        self.timeEmbedding = None
        self.ogrinal_data = list()
        self.orginal_trace = list()
        self.encode_trace = list()
        self.train_dataset = list()
        self.test_dataset = list()
        self.train_mixLengthData = list()
        self.test_mixLengthData = list()
        self.event2id = dict()
        self.id2event = dict()
        self.train_batch_mix = list()
        self.test_batch_mix = list()
        self.train_singleLengthData = dict()
        self.test_singleLengthData = dict()
        self.train_batch = dict()
        self.test_batch = dict()
        self.train_batch_single = dict()
        self.test_batch_single = dict()

        self.vocab_size = 0
        self.train_maxLength = 0
        self.test_maxLength = 0
        self.embd_dimension = embd_dimension
        self.initData(data_address)
    # 构建id2event event2id
    def initData(self, data_address):
        # print("in initData")
        id2event = dict()
        event2id = dict()
        orginal_trace = list()
        record = list()
        trace_temp = list()
        with open(data_address, 'r', encoding='utf-8') as f:
            # 数据第一行为表头
            # f = data_address
            next(f) # 跳过表头
            lines = f.readlines()
            for line in lines:
                record.append(line)
        flag = record[0].split(',')[0]
        for line in record:
            # print(line)
            line = line.replace('\r', '').replace('\n', '')
            line = line.split(',')
            # 构造id2event and event2id
            if line[1] not in event2id.keys():
                index = len(event2id)+1
                id2event[index] = line[1]
                event2id[line[1]] = index
            if line[0] == flag:
                trace_temp.append([line[1], line[2],line[3]])
            else:
                flag = line[0]
                if len(trace_temp) > 0:
                    orginal_trace.append(trace_temp.copy())
                trace_temp = list()
                trace_temp.append([line[1], line[2],line[3]])
        self.id2event = id2event
        self.event2id = event2id
        self.vocab_size = len(self.event2id)
        self.ogrinal_data = record
        #CaseID = 2 ：[['1_2', '2012-04-03 16:55:38'], ['8_8', '2012-04-03 16:55:53'], ['6_0', '2012-04-05 17:15:52']]
        self.orginal_trace = orginal_trace    #  orginal_trace是包含case，活动名和时间戳的trace的集合[[['A','timestamp'],['B','timestamp']],[['A','timestamp'],['B','timestamp'],['C','timestamp']]]
        # 生成中间文件
        # print(0)

    # 加载预训练的词向量
    def encodeEvent(self):
        event2id = dict()
        id2event = dict()

        for line in self.ogrinal_data:
            line = line.replace('\r', '').replace('\n', '')
            line = line.split(',')
            try:
                event2id[line[1]] = event2id[line[1]]
                id2event[event2id[line[1]]] = id2event[event2id[line[1]]]
            except KeyError as ke:
                event2id[line[1]] = len(event2id)
                id2event[len(id2event)] = line[1]
                # event2id[line[1]] = line[1]
                # id2event[line[1]] = line[1]
        self.vocab_size = len(event2id)
        self.embedding = nn.Embedding(self.vocab_size + 1, self.embd_dimension, padding_idx=self.vocab_size)
        # print(0)
    def encodeTrace(self):
        encode_trace = list()
        max = 0
        for line in self.orginal_trace:
            trace_temp = list()
            for line2 in line:
                trace_temp.append([self.event2id[line2[0]], line2[1],line2[2]])
            if len(trace_temp) > max:
                max = len(trace_temp)
            encode_trace.append(trace_temp.copy())
        self.max = max
        self.encode_trace = encode_trace#[[[0,'timestamp'],[1,'timestamp']],[[0,'timestamp'],[1,'timestamp'],[2,'timestamp']]]
        # print(0)
    # 分割训练集和测试集 7:3
    def splitData(self, train_splitThreshold=0.7):
        # print(len(self.encode_trace))
        self.train_dataset, self.test_dataset = train_test_split(self.encode_trace, train_size=train_splitThreshold,
                                                                 test_size=1 - train_splitThreshold, random_state=123)
        # print('111')
    # 处理train or test 生成相应长度single以及mix的list
    def initBatchData_Prefix(self):     # 训练集和测试集分别生成前缀集和后缀集

        train_singlePrefixData = []
        train_singlePrefixData1 = []
        train_labelPrefixData = []

        test_singlePrefixData = []
        test_singlePrefixData1 = []
        test_labelPrefixData = []

        train_maxLength = 0
        test_maxLength = 0

        for trace_train in self.train_dataset:#trace_train:[[0,'timestamp'],[1,'timestamp']]

            train_prefix_temp = list()
            train_labelprefix_temp=list()
            if trace_train[0][2]=='1':
                for index,line_event in enumerate(trace_train):#line_event:[0,'timestamp']
                    train_prefix_temp.append(line_event[0])   # 每条前缀的生成& len(train_prefix_temp)< len(train_prefix_temp) + len(train_suffix_temp)
                    train_labelprefix_temp.append(line_event[2])
                    # target_label = []
                    # target_currentfinal_activity = []
                    if len(train_prefix_temp) > train_maxLength:
                        train_maxLength = len(train_prefix_temp)
                    # target_currentfinal_activity.append(line_event[0])
                    if len(train_prefix_temp) > 0 and len(train_prefix_temp) <= len(trace_train): # len(train_prefix_temp) > 2 保证了前缀的长度>=3。  # len(train_prefix_temp) > 0保证了前缀的长度>=1。
                        train_singlePrefixData.append(train_prefix_temp.copy())
                    # if len(train_prefix_temp) > 1:
                        train_labelPrefixData.append([int(train_labelprefix_temp[-1])])
            else:
                for train_index,train_temp in enumerate(trace_train):
                    train_prefix_temp.append(train_temp[0])
                train_singlePrefixData.append(train_prefix_temp)
                train_labelPrefixData.append([0])
        # print(0)
        train_singlePrefixData_processing = []  # #  #  #  #  #  # #  #  #  #  #  #  #
        for each in train_singlePrefixData:  # #  #  #  #  #  # #  #  #  #  #  #  #
            train_singlePrefixData_processing.append(each[-1])  # #  #  #  #  #  # #  #  #  #  #  #  #
        # print('train_singlePrefixData_processing:   ', train_singlePrefixData_processing)  # #  #  #  #  #  # #  #  #  #  #  #  #
        # print(0)
        torch.save(train_singlePrefixData_processing, "label_train_RNN_result1.pth")

        for trace_test in self.test_dataset:
            test_prefix_temp = list()
            test_labelprefix_temp = list()
            if trace_test[0][2]=='1':
                for index, line_event in enumerate(trace_test):
                    test_prefix_temp.append(line_event[0])  # 每条前缀的生成
                    test_labelprefix_temp.append(line_event[2])
                    if len(test_prefix_temp) > test_maxLength:
                        test_maxLength = len(test_prefix_temp)

                    if len(test_prefix_temp) > 0 and len(test_prefix_temp) <= len(trace_test):# len(test_prefix_temp) > 2 保证了前缀的长度>=3。  # len(test_prefix_temp) > 0保证了前缀的长度>=1。
                            test_singlePrefixData.append(test_prefix_temp.copy())

                            test_labelPrefixData.append([int(test_labelprefix_temp[-1])])
            else:
                for train_index,train_temp in enumerate(trace_test):
                    test_prefix_temp.append(train_temp[0])
                test_singlePrefixData.append(test_prefix_temp)
                test_labelPrefixData.append([0])
        # print(0)
        test_singlePrefixData_processing = []  # #  #  #  #  #  # #  #  #  #  #  #  #
        for each in test_singlePrefixData:  # #  #  #  #  #  # #  #  #  #  #  #  #
            test_singlePrefixData_processing.append(each[-1])  # #  #  #  #  #  # #  #  #  #  #  #  #
        # print('test_singlePrefixData_processing:   ', test_singlePrefixData_processing)  # #  #  #  #  #  # #  #  #  #  #  #  #
        # print(0)
        torch.save(test_singlePrefixData_processing, "label_test_RNN_result1.pth")


        self.train_singlePrefixData = train_singlePrefixData
        self.train_labelPrefixData = train_labelPrefixData

        self.test_singlePrefixData = test_singlePrefixData
        self.test_labelPrefixData = test_labelPrefixData


        length = [len(sublist) for sublist in self.train_singlePrefixData]#sublist:[0,1,4,1,4]#length[5]
        # print("每个序列的长度为：\n", length)
        length1 = [len(sublist) for sublist in self.test_singlePrefixData]
        maxLen_trainprefix = max(length)
        maxLen_testprefix=max(length1)
        maxLen_final=max(maxLen_trainprefix,maxLen_testprefix)
        train_singlePrefixData1= [sublist + [0] * (maxLen_final - len(sublist)) for sublist in train_singlePrefixData]




        # maxLen_testprefix = max(length1)
        test_singlePrefixData1 = [sublist + [0] * (maxLen_final - len(sublist)) for sublist in
                                            test_singlePrefixData]
        # print('111')
        # print('len(train_labelPrefixData): ', len(train_labelPrefixData))    #  #  #  #  #  #  #
        # print('len(test_labelPrefixData): ', len(test_singlePrefixData))#  #  #  #  #  #  #

        return train_singlePrefixData1, train_labelPrefixData, test_singlePrefixData1, test_labelPrefixData,\
               train_singlePrefixData_processing, test_singlePrefixData_processing


#
# if __name__ == '__main__':
#     data_address = "/Users/caorui/Desktop/code5-NextActivity_NonPartition/Data/helpdesk1.csv"
#     train_splitThreshold = 0.7
#     data = InputData(data_address, embd_dimension=2)
#     # # 构建embedding
#     data.encodeEvent()
#     data.encodeTrace()
#     data.splitData()
#     data.initBatchData_Prefix()
#     # data.generateMixLengthBatch(9)