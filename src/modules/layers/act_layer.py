import torch as th
import torch.nn as nn

class ActivateLayer(nn.Module):
    def __init__(self, dim, name):
        super(ActivateLayer, self).__init__()
        self.weight = nn.Parameter(th.ones(dim))
        self.weight.requires_grad = False
        self.name = name

    def forward(self, x):
        # print('pre',x)
        # avg = x.mean()
        # x[x<0.01*avg] = 0
        # print('after',x)
        return x*self.weight
