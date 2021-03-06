import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class PDENetLight(nn.Module):

    # đọc paper để biết thêm chi tiết về tham số :))))
    # hidden_sz = M trong paper
    def __init__(self, dim_in, hidden_sz):
        super().__init__()
        self.hidden_sz = hidden_sz
        self.W1 = nn.Parameter(torch.Tensor(hidden_sz, dim_in))
        self.b1 = nn.Parameter(torch.Tensor(hidden_sz, 1))

        # 1st layer
        self.Uz1 = torch.nn.Parameter(torch.Tensor(hidden_sz, dim_in))
        self.Wz1 = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.bz1 = torch.nn.Parameter(torch.Tensor(hidden_sz, 1))

        self.Ug1 = torch.nn.Parameter(torch.Tensor(hidden_sz, dim_in))
        self.Wg1 = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.bg1 = torch.nn.Parameter(torch.Tensor(hidden_sz, 1))

        self.Ur1 = torch.nn.Parameter(torch.Tensor(hidden_sz, dim_in))
        self.Wr1 = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.br1 = torch.nn.Parameter(torch.Tensor(hidden_sz, 1))

        self.Uh1 = torch.nn.Parameter(torch.Tensor(hidden_sz, dim_in))
        self.Wh1 = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.bh1 = torch.nn.Parameter(torch.Tensor(hidden_sz, 1))

        # 2nd layer

        self.Uz2 = torch.nn.Parameter(torch.Tensor(hidden_sz, dim_in))
        self.Wz2 = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.bz2 = torch.nn.Parameter(torch.Tensor(hidden_sz, 1))

        self.Ug2 = torch.nn.Parameter(torch.Tensor(hidden_sz, dim_in))
        self.Wg2 = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.bg2 = torch.nn.Parameter(torch.Tensor(hidden_sz, 1))

        self.Ur2 = torch.nn.Parameter(torch.Tensor(hidden_sz, dim_in))
        self.Wr2 = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.br2 = torch.nn.Parameter(torch.Tensor(hidden_sz, 1))

        self.Uh2 = torch.nn.Parameter(torch.Tensor(hidden_sz, dim_in))
        self.Wh2 = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.bh2 = torch.nn.Parameter(torch.Tensor(hidden_sz, 1))

        # 3rd layer


        # output layer
        self.W = torch.nn.Parameter(torch.Tensor(1, hidden_sz))
        self.b = torch.nn.Parameter(torch.Tensor(1, 1))

    def init_weights(self):
        for p in self.parameters():
            nn.init.xavier_uniform_(p.data)

    def forward(self, x):
        Ones = torch.ones(self.hidden_sz, 1)

        S1 = torch.tanh(torch.mm(self.W1, x) + self.b1)

        # 1st layer
        Z1 = torch.tanh(torch.mm(self.Uz1, x) + torch.mm(self.Wz1, S1) + self.bz1)
        G1 = torch.tanh(torch.mm(self.Ug1, x) + torch.mm(self.Wg1, S1) + self.bg1)
        R1 = torch.tanh(torch.mm(self.Ur1, x) + torch.mm(self.Wr1, S1) + self.br1)
        # * is element - wise multiplication
        H1 = torch.tanh(torch.mm(self.Uh1, x) + torch.mm(self.Wh1, S1 * R1) + self.bh1)

        S2 = (Ones - G1) * H1 + Z1 * S1

        # 2nd layer
        Z2 = torch.tanh(torch.mm(self.Uz2, x) + torch.mm(self.Wz2, S2) + self.bz2)
        G2 = torch.tanh(torch.mm(self.Ug2, x) + torch.mm(self.Wg2, S2) + self.bg2)
        R2 = torch.tanh(torch.mm(self.Ur2, x) + torch.mm(self.Wr2, S2) + self.br2)
        H2 = torch.tanh(torch.mm(self.Uh2, x) + torch.mm(self.Wh2, S2 * R2) + self.bh2)

        S3 = (Ones - G2) * H2 + Z2 * S2


        # output layer
        out = torch.mm(self.W, S3) + self.b

        return out

    def save(self, file):
        torch.save(self, file)


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param

