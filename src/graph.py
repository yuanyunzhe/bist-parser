# -*- coding：utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ParseForest, read_conll, write_conll
from operator import itemgetter
from itertools import chain
from extend import *

import utils, time, random
import numpy as np
import shutil

import decoder
import torch.autograd as autograd

class GraphModel(DependencyModel):
    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos, lstm_shared):
        DependencyModel.__init__(self, vocab, pos, rels, enum_word, options, onto, cpos, lstm_shared)

        self.hidLayerFOH = Parameter((self.ldims * 2, self.hidden_units))
        self.hidLayerFOM = Parameter((self.ldims * 2, self.hidden_units))
        self.hidBias = Parameter((self.hidden_units))
        self.catBias = Parameter((self.hidden_units * 2))
        self.rhidLayerFOH = Parameter((2 * self.ldims, self.hidden_units))
        self.rhidLayerFOM = Parameter((2 * self.ldims, self.hidden_units))
        self.rhidBias = Parameter((self.hidden_units))
        self.rcatBias = Parameter((self.hidden_units * 2))
        if self.hidden2_units:
            self.hid2Layer = Parameter(
                (self.hidden_units * 2, self.hidden2_units))
            self.hid2Bias = Parameter((self.hidden2_units))
            self.rhid2Layer = Parameter(
                (self.hidden_units * 2, self.hidden2_units))
            self.rhid2Bias = Parameter((self.hidden2_units))
        self.outLayer = Parameter(
            (self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, 1))
        self.outBias = 0  # Parameter(1)
        self.routLayer = Parameter(
            (self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, len(self.irels)))
        self.routBias = Parameter((len(self.irels)))

        self.evl = 0
        self.ebd = 0

    def __getExpr(self, sentence, i, j, train):

        if sentence[i].headfov is None:
            sentence[i].headfov = torch.mm(sentence[i].vec, self.hidLayerFOH)

        if sentence[j].modfov is None:
            sentence[j].modfov = torch.mm(sentence[i].vec, self.hidLayerFOM)

        if self.hidden2_units > 0:
            output = torch.mm(
                self.activation(
                    self.hid2Bias +
                    torch.mm(self.activation(cat([sentence[i].headfov, sentence[j].modfov]) + self.catBias),
                             self.hid2Layer)
                ),
                self.outLayer
            ) + self.outBias

        else:
            output = torch.mm(
                self.activation(
                    sentence[i].headfov + sentence[j].modfov + self.hidBias),
                self.outLayer) + self.outBias
        return output

    def __evaluate(self, sentence, train):
        exprs = [[self.__getExpr(sentence, i, j, train)
                  for j in range(len(sentence))]
                 for i in range(len(sentence))]
        scores = np.array([[get_data(output).numpy()[0, 0]
                            for output in exprsRow] for exprsRow in exprs])
        return scores, exprs

    def __evaluateLabel(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = torch.mm(sentence[i].vec, self.rhidLayerFOH)

        if sentence[j].rmodfov is None:
            sentence[j].rmodfov = torch.mm(sentence[i].vec, self.rhidLayerFOM)

        if self.hidden2_units > 0:
            output = torch.mm(
                self.activation(
                    self.rhid2Bias +
                    torch.mm(
                        self.activation(
                            cat([sentence[i].rheadfov, sentence[j].rmodfov]) + self.rcatBias),
                        self.rhid2Layer
                    )),
                self.routLayer
            ) + self.routBias

        else:
            output = torch.mm(
                self.activation(sentence[i].rheadfov +
                                sentence[j].rmodfov + self.rhidBias),
                self.routLayer
            ) + self.routBias

        return get_data(output).numpy()[0], output[0]

    def getWordEmbeddings(self, sentences, train):
        DependencyModel.getWordEmbeddings(self, sentences, train)
        for sentence in sentences:
            for entry in sentence:
                entry.lstms = [entry.vec, entry.vec]
                entry.headfov = None
                entry.modfov = None

                entry.rheadfov = None
                entry.rmodfov = None

    def predict(self, sentences):
        self.getWordEmbeddings(sentences, False)

        for sentence in sentences:
            scores, exprs = self.__evaluate(sentence, True)
            heads = decoder.parse_proj(scores)

            for entry, head in zip(sentence, heads):
                entry.pred_parent_id = head
                entry.pred_relation = '_'

            head_list = list(heads)
            for modifier, head in enumerate(head_list[1:]):
                scores, exprs = self.__evaluateLabel(
                    sentence, head, modifier + 1)
                sentence[modifier + 1].pred_relation = self.irels[max(
                    enumerate(scores), key=itemgetter(1))[0]]

    def forward(self, sentences, errs, lerrs):
        tmp = time.time()
        self.getWordEmbeddings(sentences, True)
        self.ebd += time.time() - tmp

        for sentence in sentences:
            tmp = time.time()
            scores, exprs = self.__evaluate(sentence, True)
            self.evl += time.time() - tmp
            gold = [entry.parent_id for entry in sentence]
            heads = decoder.parse_proj(scores, gold)

            for modifier, head in enumerate(gold[1:]):
                tmp = time.time()
                rscores, rexprs = self.__evaluateLabel(sentence, head, modifier + 1)
                self.evl += time.time() - tmp
                goldLabelInd = self.rels[sentence[modifier + 1].relation]
                wrongLabelInd = \
                    max(((l, scr) for l, scr in enumerate(rscores)
                         if l != goldLabelInd), key=itemgetter(1))[0]
                if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                    lerrs += [rexprs[wrongLabelInd] - rexprs[goldLabelInd]]

        e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
        if e > 0:
            errs += [(exprs[h][i] - exprs[g][i])[0]
                     for i, (h, g) in enumerate(zip(heads, gold)) if h != g]
        return e


class Graph(DependencyParser):
    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos):
        model = GraphModel(vocab, pos, rels, enum_word, options, onto, cpos)
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.trainer = get_optim(options.optim, self.model.parameters())

    def predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP,False)):
                self.model.hid_for_1, self.model.hid_back_1, self.model.hid_for_2, self.model.hid_back_2 = [self.model.init_hidden(self.model.ldims) for _ in range(4)]
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                self.model.predict(conll_sentence)
                yield conll_sentence


    def train(self, conll_path):
        print('pytorch version:', torch.__version__)
        batch = 1
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        iSentence = 0
        start = time.time()
        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP,True))
            random.shuffle(shuffledData)
            errs = []
            lerrs = []
            for iSentence, sentence in enumerate(shuffledData):
                self.model.hid_for_1, self.model.hid_back_1, self.model.hid_for_2, self.model.hid_back_2 = [
                    self.model.init_hidden(self.model.ldims) for _ in range(4)]
                if iSentence % 100 == 0 and iSentence != 0:
                    print('Processing sentence number:', iSentence,
                          'Loss:', eloss / etotal,
                          'Errors:', (float(eerrors)) / etotal,
                          'Time', time.time() - start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0

                conll_sentence = [entry for entry in sentence if isinstance(
                    entry, utils.ConllEntry)]
                e = self.model.forward(conll_sentence, errs, lerrs)
                eerrors += e
                eloss += e
                mloss += e
                etotal += len(sentence)
                if iSentence % batch == 0 or len(errs) > 0 or len(lerrs) > 0:
                    if len(errs) > 0 or len(lerrs) > 0:
                        eerrs = torch.sum(cat(errs + lerrs))
                        eerrs.backward()
                        self.trainer.step()
                        errs = []
                        lerrs = []
                self.trainer.zero_grad()
        if len(errs) > 0:
            eerrs = (torch.sum(errs + lerrs))
            eerrs.backward()
            self.trainer.step()
        self.trainer.zero_grad()
        print("Loss: ", mloss / iSentence)
