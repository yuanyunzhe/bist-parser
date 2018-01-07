import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DependencyModel(nn.Module):
    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos, lstm_for_1, lstm_back_1):
        nn.Module.__init__(self)
        random.seed(1)

        self.gpu = options.gpu
        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu}
        self.activation = self.activations[options.activation]

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.odims = options.oembedding_dims
        self.cdims = options.cembedding_dims
        self.edims = 0
        self.external_embedding = None
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding,'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()
            self.edims = len(list(self.external_embedding.values()[0]))
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            np_emb = np.zeros((len(self.external_embedding) + 3, self.edims))
            for word, i in self.extrnd.items():
                np_emb[i] = self.external_embedding[word]
            self.elookup = nn.Embedding(*np_emb.shape)
            self.elookup.weight = Parameter(init=np_emb)
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2
            print('Load external embedding. Vector dimensions', self.edims)

        self.layers = options.lstm_layers
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in enum_word.items()}
        self.pos = {word: ind + 3 for ind, word in enumerate(pos)}
        self.onto = {word: ind + 3 for ind, word in enumerate(onto)}
        self.cpos = {word: ind + 3 for ind, word in enumerate(cpos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1
        self.onto['*PAD*'] = 1
        self.cpos['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2
        self.onto['*INITIAL*'] = 2
        self.cpos['*INITIAL*'] = 2

        self.wlookup = nn.Embedding(len(vocab) + 3, self.wdims)
        self.plookup = nn.Embedding(len(pos) + 3, self.pdims)
        self.rlookup = nn.Embedding(len(rels), self.rdims)
        self.olookup = nn.Embedding(len(onto) + 3, self.odims)
        self.clookup = nn.Embedding(len(cpos) + 3, self.cdims)

        self.lstm_for_1 = lstm_for_1
        self.lstm_back_1 = lstm_back_1
        self.lstm_for_2 = nn.LSTM(self.ldims * 2, self.ldims)
        self.lstm_back_2 = nn.LSTM(self.ldims * 2, self.ldims)
        self.hid_for_1, self.hid_back_1, self.hid_for_2, self.hid_back_2 = [self.init_hidden(self.ldims) for _ in range(4)]

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

    def init_hidden(self, dim):
        if torch.cuda.is_available():
            m = torch.zeros(1, 1, dim).cuda()
        else:
            m = torch.zeros(1, 1, dim)
        return torch.autograd.Variable(m), torch.autograd.Variable(m)

    def getWordEmbeddings(self, sentence, train):
        for entry in sentence:
            c = float(self.wordsCount.get(entry.norm, 0))
            dropFlag = not train or (random.random() < (c / (0.25 + c)))
            wordvec = self.wlookup(scalar(int(self.vocab.get(entry.norm, 0)) if dropFlag else 0)) if self.wdims > 0 else None
            posvec = self.plookup(scalar(int(self.pos[entry.pos]))) if self.pdims > 0 else None
            ontovec = self.olookup(scalar(int(self.onto[entry.onto]))) if self.odims > 0 else None
            cposvec = self.clookup(scalar(int(self.cpos[entry.cpos]))) if self.cdims > 0 else None
            if self.external_embedding is not None:
                if entry.form in self.external_embedding:
                    evec = self.elookup(scalar(self.extrnd[entry.form]))
                elif entry.norm in self.external_embedding:
                    evec = self.elookup(scalar(self.extrnd[entry.norm]))
                else:
                    evec = self.elookup(scalar(0))
            else:
                evec = None
            entry.vec = cat([wordvec, posvec, ontovec, cposvec, evec])

        num_vec = len(sentence)
        vec_for = torch.cat([entry.vec for entry in sentence]).view(num_vec, 1, -1)
        vec_back = torch.cat([entry.vec for entry in reversed(sentence)]).view(num_vec, 1, -1)
        res_for_1, hid_for_1 = self.lstm_for_1(vec_for, self.hid_for_1)
        res_back_1, hid_back_1 = self.lstm_back_1(vec_back, self.hid_back_1)
        vec_cat = [cat([res_for_1[i], res_back_1[num_vec - i - 1]]) for i in range(num_vec)]
        vec_for_2 = torch.cat(vec_cat).view(num_vec, 1, -1)
        vec_back_2 = torch.cat(list(reversed(vec_cat))).view(num_vec, 1, -1)
        res_for_2, self.hid_for_2 = self.lstm_for_2(vec_for_2, self.hid_for_2)
        res_back_2, self.hid_back_2 = self.lstm_back_2(vec_back_2, self.hid_back_2)

        for i in range(num_vec):
            lstm0 = res_for_2[i]
            lstm1 = res_back_2[num_vec - i - 1]
            sentence[i].vec = cat([lstm0, lstm1])

        self.vec = vec_cat[0].cpu().data
