import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import ParseForest, read_conll, write_conll
from operator import itemgetter
from itertools import chain
import utils, time, random

import numpy as np


class ArcHybridLSTMModel(nn.Module):
    def __init__(self, words, pos, rels, w2i, options):
        super(ArcHybridLSTM, self).__init__()

        random.seed(1)
        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu}
        self.activation = self.activations[options.activation]
        self.oracle = options.oracle
        self.ldims = options.lstm_dims * 2
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = words
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.headFlag = options.headFlag
        self.rlMostFlag = options.rlMostFlag
        self.rlFlag = options.rlFlag
        self.k = options.window

        self.nnvecs = (1 if self.headFlag else 0) + (2 if self.rlFlag or self.rlMostFlag else 0)

        self.external_embedding = None
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding,'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in range(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            np_emb = np.zeros((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                np_emb[i] = self.external_embedding[word]
            self.elookup = nn.Embedding(*np_emb.shape)
            self.elookup.weight = Parameter(np_emb)
            for word, i in self.extrnd.iteritems():
                self.elookup.init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print('Load external embedding. Vector dimensions', self.edim)

        dims = self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)
        self.blstmFlag = options.blstmFlag
        self.bibiFlag = options.bibiFlag

        if self.bibiFlag:
            self.surfaceBuilders = [nn.LSTMCell(dims, self.ldims * 0.5),
                                    nn.LSTMCell(dims, self.ldims * 0.5)]
            self.bsurfaceBuilders = [nn.LSTMCell(self.ldims, self.ldims * 0.5),
                                     nn.LSTMCell(self.ldims, self.ldims * 0.5)]
        elif self.blstmFlag:
            if self.layers > 0:
                self.surfaceBuilders = [nn.LSTMCell(self.layers, dims, self.ldims * 0.5), [nn.LSTMCell(self.layers, dims, self.ldims * 0.5)]
            else:
                self.surfaceBuilders = [nn.RNNCell(dims, self.ldims * 0.5), nn.LSTMCell(dims, self.ldims * 0.5)]

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units
        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = nn.Embedding((len(words) + 3, self.wdims))
        self.plookup = nn.Embedding((len(pos) + 3, self.pdims))
        self.rlookup = nn.Embedding((len(rels), self.rdims))

        self.word2lstm = Parameter((self.ldims, self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)))
        self.word2lstmbias = Parameter((self.ldims))
        self.lstm2lstm = Parameter((self.ldims, self.ldims * self.nnvecs + self.rdims))
        self.lstm2lstmbias = Parameter((self.ldims))

        self.hidLayer = Parameter((self.hidden_units, self.ldims * self.nnvecs * (self.k + 1)))
        self.hidBias = Parameter((self.hidden_units))

        self.hid2Layer = Parameter((self.hidden2_units, self.hidden_units))
        self.hid2Bias = Parameter((self.hidden2_units))

        self.outLayer = Parameter((3, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.outBias = Parameter((3))

        self.rhidLayer = Parameter((self.hidden_units, self.ldims * self.nnvecs * (self.k + 1)))
        self.rhidBias = Parameter((self.hidden_units))

        self.rhid2Layer = Parameter((self.hidden2_units, self.hidden_units))
        self.rhid2Bias = Parameter((self.hidden2_units))

        self.routLayer = Parameter((2 * (len(self.irels) + 0) + 1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.routBias = Parameter((2 * (len(self.irels) + 0) + 1))

    def __evaluate(self, stack, buf, train):
        topStack = [ stack.roots[-i-1].lstms if len(stack) > i else [self.empty] for i in range(self.k) ]
        topBuffer = [ buf.roots[i].lstms if len(buf) > i else [self.empty] for i in range(1) ]

        input = cat(list(chain(*(topStack + topBuffer))))

        if self.hidden2_units > 0:
            routput = torch.mm(
                self.routLayer,
                self.activation(
                    self.rhid2Bias +
                    torch.mm(
                        self.rhid2Layer,
                        self.activation(torch.mm(self.rhidLayer, input) + self.rhidBias)
                    )
                )
            ) + self.routBias
        else:
            routput = torch.mm(
                self.routLayer,
                self.activation(torch.mm(self.rhidLayer, input) + self.rhidBias)
            ) + self.routBias

        if self.hidden2_units > 0:
            output = torch.mm(
                self.outLayer,
                self.activation(
                    self.hid2Bias +
                    torch.mm(
                        self.hid2Layer,
                        self.activation(torch.mm(self.hidLayer, input) + self.hidBias)
                    )
                )
            ) + self.outBias
        else:
            output = torch.mm(
                self.outLayer,
                self.activation(torch.mm(self.hidLayer, input) + self.hidBias)
            ) + self.outBias

        scrs, uscrs = routput.numpy(), output.numpy()

        #transition conditions
        left_arc_conditions = len(stack) > 0 and len(buf) > 0
        right_arc_conditions = len(stack) > 1 and stack.roots[-1].id != 0
        shift_conditions = len(buf) >0 and buf.roots[0].id != 0

        uscrs0 = uscrs[0]
        uscrs1 = uscrs[1]
        uscrs2 = uscrs[2]
        if train:
            output0 = output[0]
            output1 = output[1]
            output2 = output[2]
            ret = [ [ (rel, 0, scrs[1 + j * 2] + uscrs1, routput[1 + j * 2 ] + output1) for j, rel in enumerate(self.irels) ] if left_arc_conditions else [],
                    [ (rel, 1, scrs[2 + j * 2] + uscrs2, routput[2 + j * 2 ] + output2) for j, rel in enumerate(self.irels) ] if right_arc_conditions else [],
                    [ (None, 2, scrs[0] + uscrs0, routput[0] + output0) ] if shift_conditions else [] ]
        else:
            s1,r1 = max(zip(scrs[1::2],self.irels))
            s2,r2 = max(zip(scrs[2::2],self.irels))
            s1 += uscrs1
            s2 += uscrs2
            ret = [ [ (r1, 0, s1) ] if left_arc_conditions else [],
                    [ (r2, 1, s2) ] if right_arc_conditions else [],
                    [ (None, 2, scrs[0] + uscrs0) ] if shift_conditions else [] ]
        return ret

    def getWordEmbeddings(self, sentence, train):
        pass

    def predict(self, conll_path):
        pass

class ArcHybridLSTM:
    def __init__(self, vocab, pos, rels, w2i, options):
        model = ArcHybridLSTMModel(vocab, pos, rels, w2i, options)
        self.model = model.cuda(options.cuda_index) if options.cuda_index >= 0 else model
        self.trainer = get_optim(options.optim, self.model.parameters())

    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.model.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.model.load_state_dict(torch.load(fn))

    def init(self):
        pass
    
    def train(self, conll_path):
        pass
