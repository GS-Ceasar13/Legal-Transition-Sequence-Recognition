
import torch
import numpy as np
import torch.nn as nn



class RNN(nn.Module):
    def __init__(self, inputsize,embedding_dim,hidden_dim,out_size,batch_size=1,n_layer = 1, dropout = 0,
                 embedding = None):
        super(RNN, self).__init__()    #   这里的5是指输入的特征数
        # self.vocab_size = vocab_size
        # self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.out_shape = out_size
        # self.embedding = embedding
        # self.batch_size = batch_size
        self.n_layer = n_layer
        self.dropout = dropout
        # self.weight_W = nn.Parameter(torch.Tensor(batch_size, hidden_dim, hidden_dim))
        # self.weight_Mu = nn.Parameter(torch.Tensor(hidden_dim, n_layer))
        self.firstLayer = nn.Sequential(nn.Linear(inputsize, 5),  # Loan_application_Configuration4
                                        # nn.ReLU(),
                                        # nn.Linear(45, 30),
                                        # nn.ReLU(),
                                        # nn.Linear(30, 5),
                                        # nn.ReLU()
                                        )
        # self.first2Layer = nn.Sequential(nn.Linear(5, 45),  # Loan_application_Configuration4
        #                                 nn.ReLU(),
        #                                 nn.Linear(45, 30),
        #                                 nn.ReLU(),
        #                                 nn.Linear(30, 5),
        #                                 nn.ReLU()
        #                                 )
        # self.rnn = nn.GRU(input_size= 1, hidden_size=hidden_dim, dropout=self.dropout,
        #                   num_layers=self.n_layer, bidirectional=False)
        # self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_dim, dropout=self.dropout,
        #                   num_layers=self.n_layer, bidirectional=False)
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_dim, dropout=self.dropout,
                          num_layers=self.n_layer, bidirectional=False)
        # self.first2Layer = nn.Sequential(nn.Linear(hidden_dim, 45),  # Loan_application_Configuration4
        #                                 nn.ReLU(),
        #                                 nn.Linear(45, 30),
        #                                 nn.ReLU(),
        #                                 nn.Linear(30, 5),
        #                                 nn.ReLU()
        #                                 )
        # self.secondLayer = nn.Sequential(nn.Linear(hidden_dim, 45),  # Loan_application_Configuration4
        #                                 nn.ReLU(),
        #                                 nn.Linear(45, 30),
        #                                 nn.ReLU(),
        #                                 nn.Linear(30, hidden_dim),
        #                                 nn.ReLU()
        #                                 )

        self.out = nn.Linear(hidden_dim, hidden_dim)



    def forward(self, x):
        first = self.firstLayer(x)
        # first = self.firstLayer(first0)
        input = first.view(5, 9, 1)
        # print(0)
        # # hidden_state = Variable(torch.randn(self.n_layer, self.batch_size, self.hidden_dim))
        # hidden_state = Variable(torch.randn(self.n_layer, self.batch_size, self.hidden_dim))
        # input1 = self.firstLayer(input)
        output0, final_hidden_state = self.rnn(input)
        # print(0)
        # s = torch.load("myfinalHideState5weiRNNhelpdesk1.pth")
        # q = final_hidden_state[-1].detach().numpy().reshape(9, 10)
        # temp = np.vstack((s, q))
        #
        # torch.save(q, "myfinalHideState5weiRNNhelpdesk1.pth")
        # torch.save(temp, "myfinalHideState5weiRNNhelpdesk1.pth")
        hn0 = output0[-1]
        output = self.out(hn0)
        # output = self.secondLayer(hn0)
        return output
