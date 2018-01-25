import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.init import *
import utils
import shutil

len_sen = lambda x: sum([1 if isinstance(i, utils.ConllEntry) else 0 for i in x])


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
    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos, lstm_shared):
        nn.Module.__init__(self)
        random.seed(1)

        self.gpu = options.gpu
        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu}
        self.activation = self.activations[options.activation]
        self.batch = options.batch

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

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

        # Network Structure
        self.lstm_shared = lstm_shared
        self.lstm_specific = nn.LSTM(self.ldims * 2, self.ldims, batch_first=True, bidirectional=True)
        # self.hid1, self.hid2 = [self.init_hidden(self.ldims) for _ in range(2)]

    def init_hidden(self, batch, dim):
        h = torch.zeros(2, batch, dim)
        c = torch.zeros(2, batch, dim)
        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()
        return torch.autograd.Variable(h), torch.autograd.Variable(c)

    def getWordEmbeddings(self, sentences, train):
        batch = len(sentences)
        for sentence in sentences:
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
            sentence[0].ebd = torch.cat([entry.vec for entry in sentence])
            # print("\t",len_sen(sentences[0]), sentence[0].ebd.shape)
        ebd = torch.cat([sentence[0].ebd for sentence in sentences]).view(batch, len_sen(sentences[0]), -1)
        h0, c0 = self.init_hidden(batch, self.ldims)
        # print(ebd.shape, h0.shape, batch, len_sen(sentences[0]), self.ldims)
        out_shared, _ = self.lstm_shared(ebd, (h0, c0))
        h1, c1 = self.init_hidden(batch, self.ldims)
        out, _ = self.lstm_specific(out_shared, (h1, c1))
        for i in range(batch):
            for j in range(len(sentences[i])):

                sentences[i][j].vec = out[i][j].view(1, -1)
        # self.vec = out_shared[0].cpu().data


class DependencyParser(object):
    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.model.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.model.load_state_dict(torch.load(fn))


# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.shared = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
#         self.graph = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
#         self.transition = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
#
#     def forward(self, *input):