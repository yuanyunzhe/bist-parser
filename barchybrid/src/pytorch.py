import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn.init import *

from utils import ParseForest, read_conll, write_conll
from operator import itemgetter
from itertools import chain
import utils, time, random

import numpy as np


def Parameter(shape=None, init=xavier_uniform):
    if hasattr(init, 'shape'):
        assert not shape
        return nn.Parameter(torch.Tensor(init))
    shape = (shape, 1) if type(shape) == int else shape
    return nn.Parameter(init(torch.Tensor(*shape)))

def scalar(f):
    if type(f) == int:
        return Variable(torch.LongTensor([f]))
    if type(f) == float:
        return Variable(torch.FloatTensor([f]))


def cat(l, dimension=-1):
    valid_l = filter(lambda x: type(x) != type(None), l)
    valid_l = list(valid_l)
    # print("_____\n",valid_l)
    if dimension < 0:
        dimension += len(valid_l[0].size())
    # print(valid_l[0])
    # print(dimension)
    return torch.cat(valid_l, dimension)


class RNNState():
    def __init__(self, cell, hidden=None):
        self.cell = cell
        self.hidden = hidden
        if not hidden:
            self.hidden = Variable(torch.zeros(1, self.cell.hidden_size)), \
                          Variable(torch.zeros(1, self.cell.hidden_size))

    def next(self, input):
        return RNNState(self.cell, self.cell(input, self.hidden))

    def __call__(self):
        return self.hidden[0]


class ArcHybridLSTMModel(nn.Module):
    def __init__(self, words, pos, rels, w2i, options):
        super(ArcHybridLSTMModel, self).__init__()

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
        self.vocab = {word: ind+3 for word, ind in w2i.items()}
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
            external_embedding_fp = open(options.external_embedding,'r', encoding='UTF-8')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in range(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            np_emb = np.zeros((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.items():
                np_emb[i] = self.external_embedding[word]
            self.elookup = nn.Embedding(*np_emb.shape)
            self.elookup.weight = Parameter(np_emb)
            for word, i in self.extrnd.items():
                self.elookup.init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print('Load external embedding. Vector dimensions', self.edim)

        dims = self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)
        self.blstmFlag = options.blstmFlag
        self.bibiFlag = options.bibiFlag

        if self.bibiFlag:
            self.builders = [nn.LSTMCell(dims, self.ldims // 2),
                                    nn.LSTMCell(dims, self.ldims // 2)]
            self.bbuilders = [nn.LSTMCell(self.ldims, self.ldims // 2),
                                     nn.LSTMCell(self.ldims, self.ldims // 2)]
        elif self.blstmFlag:
            if self.layers > 0:
                self.builders = [nn.LSTMCell(self.layers, dims, self.ldims // 2), nn.LSTMCell(self.layers, dims, self.ldims // 2)]
            else:
                self.builders = [nn.RNNCell(dims, self.ldims // 2), nn.LSTMCell(dims, self.ldims // 2)]

        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units
        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = nn.Embedding(len(words) + 3, self.wdims)
        self.plookup = nn.Embedding(len(pos) + 3, self.pdims)
        self.rlookup = nn.Embedding(len(rels), self.rdims)

        self.word2lstm = Parameter((self.ldims, self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)))
        self.word2lstmbias = Parameter((self.ldims))
        self.lstm2lstm = Parameter((self.ldims, self.ldims * self.nnvecs + self.rdims))
        self.lstm2lstmbias = Parameter((self.ldims))

        self.hidLayer = Parameter((self.hidden_units, self.ldims * self.nnvecs * (self.k + 1)))
        self.hidBias = Parameter((self.hidden_units))

        if self.hidden2_units:
            self.hid2Layer = Parameter((self.hidden2_units, self.hidden_units))
            self.hid2Bias = Parameter((self.hidden2_units))

        self.outLayer = Parameter((3, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.outBias = Parameter((3))

        self.rhidLayer = Parameter((self.hidden_units, self.ldims * self.nnvecs * (self.k + 1)))
        self.rhidBias = Parameter((self.hidden_units))

        if self.hidden2_units:
            self.rhid2Layer = Parameter((self.hidden2_units, self.hidden_units))
            self.rhid2Bias = Parameter((self.hidden2_units))

        self.routLayer = Parameter((2 * (len(self.irels) + 0) + 1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.routBias = Parameter((2 * (len(self.irels) + 0) + 1))

    def __evaluate(self, stack, buf, train):
        topStack = [ stack.roots[-i-1].lstms if len(stack) > i else [self.empty.t()] for i in range(self.k) ]
        topBuffer = [ buf.roots[i].lstms if len(buf) > i else [self.empty.t()] for i in range(1) ]

        input = chain(*(topStack + topBuffer))
        # print(self.rhidLayer.data.shape)
        # print(self.empty.data.shape, buf.roots[0].lstms[0].data.shape)
        # for i in input:
            # print("    ",i.data.shape)
        input = cat(input).t()
        # print("!!!!!!!!!!!", input, input.t())

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

        scrs, uscrs = routput.data.numpy(), output.data.numpy()

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

    def GetWordEmbeddings(self, sentence, train):
        for root in sentence:
            c = float(self.wordsCount.get(root.norm, 0))
            dropFlag =  not train or (random.random() < (c/(0.25+c)))
            root.wordvec = self.wlookup(scalar(int(self.vocab.get(root.norm, 0)) if dropFlag else 0))
            root.posvec = self.plookup(scalar(int(self.pos[root.pos]))) if self.pdims > 0 else None

            if self.external_embedding is not None:
                if root.form in self.external_embedding:
                    root.evec = self.elookup(scalar(self.extrnd[root.form]))
                elif root.norm in self.external_embedding:
                    root.evec = self.elookup(scalar(self.extrnd[root.norm]))
                else:
                    root.evec = self.elookup(scalar(0))
            else:
                root.evec = None
            root.ivec = cat([root.wordvec, root.posvec, root.evec])

        if self.blstmFlag:
            forward  = RNNState(self.builders[0])
            backward = RNNState(self.builders[1])

            for froot, rroot in zip(sentence, reversed(sentence)):
                forward = forward.next( froot.ivec )
                backward = backward.next( rroot.ivec )
                froot.fvec = forward()
                rroot.bvec = backward()
            for root in sentence:
                root.vec = cat([root.fvec, root.bvec])

            if self.bibiFlag:
                bforward  = RNNState(self.bbuilders[0])
                bbackward = RNNState(self.bbuilders[1])

                for froot, rroot in zip(sentence, reversed(sentence)):
                    bforward = bforward.next( froot.vec )
                    bbackward = bbackward.next( rroot.vec )
                    froot.bfvec = bforward()
                    rroot.bbvec = bbackward()
                for root in sentence:
                    root.vec = cat([root.bfvec, root.bbvec])

        else:
            for root in sentence:
                root.ivec = torch.nn(self.word2lstm, root.ivec) + self.word2lstmbias
                root.vec = torch.tanh( root.ivec )

    def Predict(self, sentence):
        sentence = sentence[1:] + [sentence[0]]
        self.GetWordEmbeddings(sentence, False)
        stack = ParseForest([])
        buf = ParseForest(sentence)

        for root in sentence:
            root.lstms = [root.vec for _ in range(self.nnvecs)]

        hoffset = 1 if self.headFlag else 0

        while not (len(buf) == 1 and len(stack) == 0):
            scores = self.__evaluate(stack, buf, False)
            best = max(chain(*scores), key = itemgetter(2) )

            if best[1] == 2:
                stack.roots.append(buf.roots[0])
                del buf.roots[0]

            elif best[1] == 0:
                child = stack.roots.pop()
                parent = buf.roots[0]

                child.pred_parent_id = parent.id
                child.pred_relation = best[0]

                bestOp = 0
                if self.rlMostFlag:
                    parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                if self.rlFlag:
                    parent.lstms[bestOp + hoffset] = child.vec

            elif best[1] == 1:
                child = stack.roots.pop()
                parent = stack.roots[-1]

                child.pred_parent_id = parent.id
                child.pred_relation = best[0]

                bestOp = 1
                if self.rlMostFlag:
                    parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                if self.rlFlag:
                    parent.lstms[bestOp + hoffset] = child.vec

    def Forward(self, sentence, errs, cache):
        eloss, mloss, errors, eerrors, lerrors, etotal = cache
        ninf = -np.inf

        sentence = sentence[1:] + [sentence[0]]
        self.GetWordEmbeddings(sentence, True)
        stack = ParseForest([])
        buf = ParseForest(sentence)

        for root in sentence:
            root.lstms = [root.vec for _ in range(self.nnvecs)]

        hoffset = 1 if self.headFlag else 0

        while not (len(buf) == 1 and len(stack) == 0):
            scores = self.__evaluate(stack, buf, True)
            scores.append([(None, 3, ninf ,None)])

            alpha = stack.roots[:-2] if len(stack) > 2 else []
            s1 = [stack.roots[-2]] if len(stack) > 1 else []
            s0 = [stack.roots[-1]] if len(stack) > 0 else []
            b = [buf.roots[0]] if len(buf) > 0 else []
            beta = buf.roots[1:] if len(buf) > 1 else []

            left_cost  = ( len([h for h in s1 + beta if h.id == s0[0].parent_id]) +
                            len([d for d in b + beta if d.parent_id == s0[0].id]) )  if len(scores[0]) > 0 else 1
            right_cost = ( len([h for h in b + beta if h.id == s0[0].parent_id]) +
                            len([d for d in b + beta if d.parent_id == s0[0].id]) )  if len(scores[1]) > 0 else 1
            shift_cost = ( len([h for h in s1 + alpha if h.id == b[0].parent_id]) +
                            len([d for d in s0 + s1 + alpha if d.parent_id == b[0].id]) )  if len(scores[2]) > 0 else 1
            costs = (left_cost, right_cost, shift_cost, 1)

            bestValid = max(( s for s in chain(*scores) if costs[s[1]] == 0 and ( s[1] == 2 or  s[0] == stack.roots[-1].relation ) ), key=itemgetter(2))
            bestWrong = max(( s for s in chain(*scores) if costs[s[1]] != 0 or  ( s[1] != 2 and s[0] != stack.roots[-1].relation ) ), key=itemgetter(2))
            best = bestValid if ( (not self.oracle) or (bestValid[2] - bestWrong[2] > 1.0) or (bestValid[2] > bestWrong[2] and random.random() > 0.1) ) else bestWrong

            if best[1] == 2:
                stack.roots.append(buf.roots[0])
                del buf.roots[0]

            elif best[1] == 0:
                child = stack.roots.pop()
                parent = buf.roots[0]

                child.pred_parent_id = parent.id
                child.pred_relation = best[0]

                bestOp = 0
                if self.rlMostFlag:
                    parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                if self.rlFlag:
                    parent.lstms[bestOp + hoffset] = child.vec

            elif best[1] == 1:
                child = stack.roots.pop()
                parent = stack.roots[-1]

                child.pred_parent_id = parent.id
                child.pred_relation = best[0]

                bestOp = 1
                if self.rlMostFlag:
                    parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                if self.rlFlag:
                    parent.lstms[bestOp + hoffset] = child.vec

            if bestValid[2] < bestWrong[2] + 1.0:
                loss = bestWrong[3] - bestValid[3]
                mloss += 1.0 + bestWrong[2] - bestValid[2]
                eloss += 1.0 + bestWrong[2] - bestValid[2]
                errs.append(loss)

            if best[1] != 2 and (child.pred_parent_id != child.parent_id or child.pred_relation != child.relation):
                lerrors += 1
                if child.pred_parent_id != child.parent_id:
                    errors += 1
                    eerrors += 1

            etotal += 1

        return (eloss, mloss, errors, eerrors, lerrors, etotal)
            

    def Init(self):
        evec = self.elookup(scalar(1)) if self.external_embedding is not None else None
        paddingWordVec = self.wlookup(scalar(1))
        paddingPosVec = self.plookup(scalar(1)) if self.pdims > 0 else None

        paddingVec = torch.tanh(torch.mm(
            self.word2lstm,
            cat([paddingWordVec, paddingPosVec, evec]).t()
            ) 
            + self.word2lstmbias)
        self.empty = paddingVec if self.nnvecs == 1 else cat([paddingVec for _ in range(self.nnvecs)])


def get_optim(opt, parameters):
    if opt == 'sgd':
        return optim.SGD(parameters, lr=opt.lr)
    elif opt == 'adam':
        return optim.Adam(parameters)


class ArcHybridLSTM:
    def __init__(self, vocab, pos, rels, w2i, options):
        model = ArcHybridLSTMModel(vocab, pos, rels, w2i, options)
        self.model = model.cuda(options.cuda_index) if options.cuda_index >= 0 else model
        self.trainer = get_optim(options.optim, self.model.parameters())
        self.headFlag = options.headFlag
        # self.external_embedding = self.model.external_embedding

    def Save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.model.state_dict(), tmp)
        shutil.move(tmp, fn)

    def Load(self, fn):
        self.model.load_state_dict(torch.load(fn))

    def Predict(self, conll_path):
        self.model.Init()
        with open(conll_path, 'r', encoding='UTF-8') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, False)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                self.model.Predict(conll_sentence)
                yield sentence

    def Train(self, conll_path):
        mloss = 0.0
        errors = 0
        batch = 0
        eloss = 0.0
        eerrors = 0
        lerrors = 0
        etotal = 0        

        hoffset = 1 if self.headFlag else 0

        start = time.time()

        with open(conll_path, 'r', encoding='UTF-8') as conllFP:
            shuffledData = list(read_conll(conllFP, True))
            random.shuffle(shuffledData)

            errs = []
            eeloss = 0.0
            self.model.Init()

            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 100 == 0 and iSentence != 0:
                    print('Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Labeled Errors:', (float(lerrors) / etotal) , 'Time', time.time()-start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0

                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                
                cache = (eloss, mloss, errors, eerrors, lerrors, etotal)
                eloss, mloss, errors, eerrors, lerrors, etotal = self.model.Forward(conll_sentence, errs, cache)

                if len(errs) > 50: # or True:
                    eerrs = torch.sum(cat(errs))
                    eerrs.backward()
                    self.trainer.step()
                    errs = []

                    self.model.Init()

        if len(errs) > 0:
            eerrs = torch.sum(cat(errs)) # * (1.0/(float(len(errs))))
            eerrs.backward()
            self.trainer.step()

            errs = []

        self.trainer.zero_grad()
        print("Loss: ", mloss/iSentence)
