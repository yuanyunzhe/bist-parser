import torch
import torch.nn as nn
from torch import optim
from torch.nn.init import *

def get_data(x):
    if torch.cuda.is_available():
        return x.data.cpu()
    else:
        return x.data


def scalar(f):
    if type(f) == int:
        return Variable(torch.LongTensor([f]))
    if type(f) == float:
        return Variable(torch.FloatTensor([f]))


def Parameter(shape=None, init=xavier_uniform):
    if hasattr(init, 'shape'):
        assert not shape
        return nn.Parameter(torch.Tensor(init))
    shape = (1, shape) if type(shape) == int else shape
    return nn.Parameter(init(torch.Tensor(*shape)))


def Variable(inner):
    if torch.cuda.is_available():
        return torch.autograd.Variable(inner.cuda())
    else:
        return torch.autograd.Variable(inner)


def cat(l, dimension=-1):
    valid_l = [x for x in l if x is not None]
    if dimension < 0:
        dimension += len(valid_l[0].size())
    return torch.cat(valid_l, dimension)


def get_optim(opt, parameters):
    if opt == 'sgd':
        return optim.SGD(parameters, lr=opt.lr)
    elif opt == 'adam':
        return optim.Adam(parameters)
