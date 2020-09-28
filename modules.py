import torch
import torch.nn as nn
import math

class SA(nn.Module):
    def __init__(self, ch, head):
        super().__init__()
        self.ch = ch
        self.head = head
        self.squeeze = ch//10
        self.dk = self.squeeze//head
        self.q = nn.Conv2d(ch, self.squeeze, 1)
        self.k = nn.Conv2d(ch, self.squeeze, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.integrate = nn.Conv2d(head, 1, 1)
        self.softmax = nn.Softmax(dim=-1)

        self.f = nn.Sequential(nn.Conv2d(ch, 1, 1),
                               nn.Sigmoid())       
    def forward(self, input):
        bs, h = input.shape[0], input.shape[2]
        #input: bs, ch, h, w
        #input: feature map
        o = self.f(input)
        return o

        """
        query = self.q(input) #bs, ch//10, h, w
        key = self.k(input)
        value = self.v(input)

        query = query.view(bs, self.head, self.dk, -1).permute(0,1,3,2) #bs, head, h*w, ch//head
        key = key.view(bs, self.head, self.dk, -1) #bs, head, ch//head, h*w
        query = query/(torch.sqrt(torch.norm(query, dim=3, keepdim=True))+1e-12)
        key = key/(torch.sqrt(torch.norm(key, dim=2, keepdim=True))+1e-12)
        s = torch.matmul(query, key) #bs, head, h*w, h*w
        #output = s.view(bs, self.head*h*h, h, h)
        if self.head > 1:
            s = self.integrate(s).squeeze(1)
        else:
            s = s.squeeze(1)
        self.s = s

        attention = self.softmax(s).permute(0,2,1) #torch.softmax(s/math.sqrt(self.dk), dim=1) #bs, h*w, h*w
        value = value.view(bs, self.ch, -1) #bs, ch, h*w
        o = torch.matmul(value, attention).view(bs, self.ch, h, -1) #bs, ch, h, w
        """
        return o

class CA(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        r = 10
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(ch, ch//r)
        self.bn = nn.BatchNorm1d(ch//r)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ch//r, ch)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, input):
        """
        b_size, h = input.shape[0], input.shape[2]
        query = input.clone().view(b_size, self.ch, -1) #b,c,h*w
        query = query/(torch.sqrt(torch.norm(query, dim=2, keepdim=True))+1e-12)
        key = input.clone().view(b_size, self.ch, -1)
        key = key/(torch.sqrt(torch.norm(key, dim=2, keepdim=True))+1e-12)
        value = input.clone().view(b_size, self.ch, -1)
        energy = torch.bmm(query, key.permute(0,2,1)) #b,c,c
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        r = torch.bmm(attention, value) #b,c,h*w
        r = r.view(b_size, self.ch, h, -1)
        """
        bs = input.shape[0]
        r = self.gap(input).view(bs, self.ch)
        r = self.fc1(r)
        r = self.bn(r)
        r = self.relu(r)
        r = self.fc2(r)
        r = torch.sigmoid(r)
        r = r.view(bs, self.ch, 1, 1)

        return r

class TAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.tensor([0.05])
        self.one = torch.tensor([1.0])
        self.zero = torch.tensor([0.0])
        self._parameters['theta']=self.theta
        self._parameters["one"]=self.one
        self._parameters["zero"]=self.zero
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, input):
        f_max = self.gmp(input)
        f = input/(f_max+1e-10)
        f_ones = torch.where(f > self.theta, self.one, self.zero)
        feature_map = self.gap(f_ones*input)/(self.gap(f_ones)+1e-10)
        return feature_map

