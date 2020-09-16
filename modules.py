import torch
import torch.nn as nn

class SA(nn.Module):
    def __init__(self, i_ch):
        super().__init__()
        self.i_ch = i_ch
        self.o_ch = 1000
        self.q = nn.Conv2d(i_ch, self.o_ch, 1)
        self.k = nn.Conv2d(i_ch, self.o_ch, 1)
        self.v = nn.Conv2d(i_ch, i_ch, 1)

    def forward(self, input):
        b_size, h = input.shape[0], input.shape[2]

        #input is feature map
        query = self.q(input.clone())
        key = self.k(input.clone())
        value = self.v(input.clone())

        query = query.view(b_size, self.o_ch, -1).permute(0,2,1)
        key = key.view(b_size, self.o_ch, -1)
        s = torch.bmm(query, key)

        alpha = torch.sigmoid(s)
        value = value.view(b_size, self.i_ch, -1)
        o = torch.bmm(value, alpha)
        o = o.view(b_size, self.i_ch, h, -1)
        return o

class CA(nn.Module):
    def __init__(self, i_ch):
        super().__init__()
        self.i_ch = i_ch
    def forward(self, input):
        b_size, h = input.shape[0], input.shape[2]
        query = input.clone().view(b_size, self.i_ch, -1) #b,c,h*w
        key = input.clone().view(b_size, self.i_ch, -1)
        value = input.clone().view(b_size, self.i_ch, -1)
        gamma = torch.sigmoid(torch.bmm(query, key.permute(0,2,1))) #b,c,c
        r = torch.bmm(value.permute(0,2,1), gamma) #b,h*w,c
        r = r.permute(0,2,1) #b,c,h*w
        r = r.view(b_size, self.i_ch, h, -1)
        return r

