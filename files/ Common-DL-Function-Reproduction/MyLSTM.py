import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter


class LSTMcell(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size

        self.W_xi=nn.Linear(input_size,hidden_size)
        self.W_xf=nn.Linear(input_size,hidden_size)
        self.W_xo=nn.Linear(input_size,hidden_size)
        self.W_xc=nn.Linear(input_size,hidden_size)

        self.W_hi=nn.Linear(hidden_size,hidden_size)
        self.W_hf=nn.Linear(hidden_size,hidden_size)
        self.W_ho=nn.Linear(hidden_size,hidden_size)
        self.W_hc=nn.Linear(hidden_size,hidden_size)

        self.W_q=nn.Linear(hidden_size,output_size)


    def forward(self,X,H_0,C_0):
        i=torch.sigmoid(self.W_xi(X)+self.W_hi(H_0))        # input_gate
        f=torch.sigmoid(self.W_xf(X)+self.W_hf(H_0))        # forget_gate
        o=torch.sigmoid(self.W_xo(X)+self.W_ho(H_0))        # output_gate
        c=torch.tanh(self.W_xc(X)+self.W_hc(H_0))           # candidate memory cell

        C_1=f*C_0+i*c
        H_1=o*torch.tanh(C_1)
        Y=self.W_q(H_1)

        return H_1,C_1,Y


class MyLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.cell=LSTMcell(input_size,hidden_size,output_size)

    def forward(self,X):
        batch_size,seq_length,_=X.size()    # x:(batch_size, seq_length, input_size),input_size表示词元embedding后的向量维度（区别于seq_length表示输入序列长度）
        h_t=torch.zeros(batch_size,self.hidden_size)
        c_t=torch.zeros(batch_size,self.hidden_size)
        output=[]

        for i in range(seq_length):
            x_t=X[:,i,:]
            h_tt,c_tt,y_t=self.cell(x_t,h_t,c_t)
            output.append(y_t.unsqueeze(1))
            c_t=c_tt
            h_t=h_tt

        return torch.cat(output, dim=1),h_tt,c_tt



# 检查张量维度是否匹配

# 超参数
input_size = 10
hidden_size = 20
output_size=16
seq_len = 5
batch_size = 3

# 输入数据
x = torch.randn(batch_size, seq_len, input_size)

# 模型
model = MyLSTM(input_size, hidden_size, output_size)

# 前向传播
output, h_n, c_n = model(x)

print(output.shape)  # (batch, seq_len, hidden_size)
print(h_n.shape)     # (batch, hidden_size)
print(c_n.shape)     # (batch, hidden_size)


        