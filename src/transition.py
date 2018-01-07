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


class TransitionModel(DependencyModel):
    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos, lstm_for_1, lstm_back_1):
        DependencyModel.__init__(self, vocab, pos, rels, enum_word, options, onto, cpos, lstm_for_1, lstm_back_1)

        self.oracle = options.oracle
        self.headFlag = options.headFlag
        self.rlMostFlag = options.rlMostFlag
        self.rlFlag = options.rlFlag

        self.k = options.window
        self.nnvecs = (1 if self.headFlag else 0) + (2 if self.rlFlag or self.rlMostFlag else 0)

        dims = self.wdims + self.pdims + self.edims
        self.word2lstm = Parameter((dims, self.ldims * 2))
        self.word2lstmbias = Parameter((self.ldims * 2))
        self.lstm2lstm = Parameter((self.ldims * 2 * self.nnvecs + self.rdims, self.ldims))
        self.lstm2lstmbias = Parameter((self.ldims * 2))

        self.hidLayer = Parameter((self.ldims * 2 * self.nnvecs * (self.k + 1), self.hidden_units))
        self.hidBias = Parameter((self.hidden_units))
        self.outLayer = Parameter((self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, 3))
        self.outBias = Parameter((3))
        self.rhidLayer = Parameter((self.ldims * 2 * self.nnvecs * (self.k + 1), self.hidden_units))
        self.rhidBias = Parameter((self.hidden_units))
        self.routLayer = Parameter((self.hidden2_units if self.hidden2_units > 0 else self.hidden_units, 2 * (len(self.irels) + 0) + 1))
        self.routBias = Parameter((2 * (len(self.irels) + 0) + 1))
        if self.hidden2_units:
            self.hid2Layer = Parameter((self.hidden_units, self.hidden2_units))
            self.hid2Bias = Parameter((self.hidden2_units))
            self.rhid2Layer = Parameter((self.hidden_units, self.hidden2_units))
            self.rhid2Bias = Parameter((self.hidden2_units))

    def __evaluate(self, stack, buf, train):
        topStack = [ stack.roots[-i-1].lstms if len(stack) > i else [self.empty] for i in range(self.k) ]
        topBuffer = [ buf.roots[i].lstms if len(buf) > i else [self.empty] for i in range(1) ]
        input = cat(chain(*(topStack + topBuffer)))

        if self.hidden2_units > 0:
            routput = torch.mm(
                self.activation(
                    self.rhid2Bias +
                    torch.mm(
                        self.activation(torch.mm(input, self.rhidLayer) + self.rhidBias),
                        self.rhid2Layer
                    )
                ),
                self.routLayer
            ) + self.routBias
        else:
            routput = torch.mm(
                self.activation(torch.mm(input, self.rhidLayer) + self.rhidBias),
                self.routLayer
            ) + self.routBias
        if self.hidden2_units > 0:
            output = torch.mm(
                self.activation(
                    self.hid2Bias +
                    torch.mm(
                        self.activation(torch.mm(input, self.hidLayer) + self.hidBias),
                        self.hid2Layer
                    )
                ),
                self.outLayer
            ) + self.outBias
        else:
            output = torch.mm(
                self.activation(torch.mm(input, self.hidLayer) + self.hidBias),
                self.outLayer
            ) + self.outBias

        scrs, uscrs = get_data(routput), get_data(output)
        scrs = scrs[0]
        uscrs = uscrs[0]
        #transition conditions
        left_arc_conditions = len(stack) > 0 and len(buf) > 0
        right_arc_conditions = len(stack) > 1 and stack.roots[-1].id != 0
        shift_conditions = len(buf) >0 and buf.roots[0].id != 0
        uscrs0 = uscrs[0]
        uscrs1 = uscrs[1]
        uscrs2 = uscrs[2]
        if train:
            output, routput = output.t(), routput.t()
            output0 = output[0]
            output1 = output[1]
            output2 = output[2]
            ret = [[(rel, 0, scrs[1 + j * 2] + uscrs1, routput[1 + j * 2 ] + output1) for j, rel in enumerate(self.irels)] if left_arc_conditions else [],
                   [(rel, 1, scrs[2 + j * 2] + uscrs2, routput[2 + j * 2 ] + output2) for j, rel in enumerate(self.irels)] if right_arc_conditions else [],
                   [(None, 2, scrs[0] + uscrs0, routput[0] + output0)] if shift_conditions else []]
        else:
            s1, r1 = max(zip(scrs[1::2],self.irels))
            s2, r2 = max(zip(scrs[2::2],self.irels))
            s1 += uscrs1
            s2 += uscrs2
            ret = [[(r1, 0, s1)] if left_arc_conditions else [],
                   [(r2, 1, s2)] if right_arc_conditions else [],
                   [(None, 2, scrs[0] + uscrs0)] if shift_conditions else []]
        return ret

    def getWordEmbeddings(self, sentence, train):
        DependencyModel.getWordEmbeddings(self, sentence, train)

    def predict(self, sentence):
        self.getWordEmbeddings(sentence, False)

        stack = ParseForest([])
        buf = ParseForest(sentence)
        for root in sentence:
            root.lstms = [root.vec for _ in range(self.nnvecs)]
        hoffset = 1 if self.headFlag else 0

        while not (len(buf) == 1 and len(stack) == 0):
            scores = self.__evaluate(stack, buf, False)
            best = max(chain(*scores), key=itemgetter(2))
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

    def forward(self, sentence, errs):
        self.getWordEmbeddings(sentence, True)

        dloss, deerrors, dlerrors, detotal = 0, 0, 0, 0
        stack = ParseForest([])
        buf = ParseForest(sentence)
        for root in sentence:
            root.lstms = [root.vec for _ in range(self.nnvecs)]
        hoffset = 1 if self.headFlag else 0
        while not (len(buf) == 1 and len(stack) == 0):
            scores = self.__evaluate(stack, buf, True)
            scores.append([(None, 3, -np.inf ,None)])
            alpha = stack.roots[:-2] if len(stack) > 2 else []
            s1 = [stack.roots[-2]] if len(stack) > 1 else []
            s0 = [stack.roots[-1]] if len(stack) > 0 else []
            b = [buf.roots[0]] if len(buf) > 0 else []
            beta = buf.roots[1:] if len(buf) > 1 else []
            left_cost  = (len([h for h in s1 + beta if h.id == s0[0].parent_id]) +
                          len([d for d in b + beta if d.parent_id == s0[0].id])) if len(scores[0]) > 0 else 1
            right_cost = (len([h for h in b + beta if h.id == s0[0].parent_id]) +
                          len([d for d in b + beta if d.parent_id == s0[0].id])) if len(scores[1]) > 0 else 1
            shift_cost = (len([h for h in s1 + alpha if h.id == b[0].parent_id]) +
                          len([d for d in s0 + s1 + alpha if d.parent_id == b[0].id])) if len(scores[2]) > 0 else 1
            costs = (left_cost, right_cost, shift_cost, 1)
            bestValid = max((s for s in chain(*scores) if costs[s[1]] == 0 and ( s[1] == 2 or  s[0] == stack.roots[-1].relation)), key=itemgetter(2))
            bestWrong = max((s for s in chain(*scores) if costs[s[1]] != 0 or  ( s[1] != 2 and s[0] != stack.roots[-1].relation)), key=itemgetter(2))
            best = bestValid if ((not self.oracle) or (bestValid[2] - bestWrong[2] > 1.0) or (bestValid[2] > bestWrong[2] and random.random() > 0.1)) else bestWrong
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
                dloss += 1.0 + bestWrong[2] - bestValid[2]
                errs.append(loss)
            if best[1] != 2 and (child.pred_parent_id != child.parent_id or child.pred_relation != child.relation):
                dlerrors += 1
                if child.pred_parent_id != child.parent_id:
                    deerrors += 1
            detotal += 1
        return dloss, deerrors, dlerrors, detotal

    def init(self):
        evec = self.elookup(scalar(1)) if self.external_embedding is not None else None
        paddingWordVec = self.wlookup(scalar(1))
        paddingPosVec = self.plookup(scalar(1)) if self.pdims > 0 else None
        paddingVec = torch.tanh(torch.mm(cat([paddingWordVec, paddingPosVec, evec]), self.word2lstm) + self.word2lstmbias)
        self.empty = paddingVec if self.nnvecs == 1 else cat([paddingVec for _ in range(self.nnvecs)])


class Transition:
    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos):
        model = TransitionModel(vocab, pos, rels, enum_word, options, onto, cpos)
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.trainer = get_optim(options.optim, self.model.parameters())
        self.headFlag = options.headFlag
        self.gpu = options.gpu
        # self.external_embedding = self.model.external_embedding

    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.model.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.model.load_state_dict(torch.load(fn))

    def predict(self, conll_path):
        self.model.init()
        with open(conll_path, 'r', encoding='UTF-8') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, proj=False)):
                self.model.hid_for_1, self.model.hid_back_1, self.model.hid_for_2, self.model.hid_back_2 = [self.model.init_hidden(self.model.ldims) for _ in range(4)]
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                conll_sentence = conll_sentence[1:] + [conll_sentence[0]]
                self.model.predict(conll_sentence)
                self.trainer.zero_grad()
                yield sentence

    def train(self, conll_path):
        mloss = 0.0
        batch = 0
        eloss = 0.0
        eerrors = 0
        lerrors = 0
        etotal = 0
        hoffset = 1 if self.headFlag else 0
        start = time.time()

        with open(conll_path, 'r', encoding='UTF-8') as conllFP:
            shuffledData = list(read_conll(conllFP, proj=True))
            random.shuffle(shuffledData)
            errs = []
            eeloss = 0.0
            self.model.init()
            for iSentence, sentence in enumerate(shuffledData):
                self.model.hid_for_1, self.model.hid_back_1, self.model.hid_for_2, self.model.hid_back_2 = [self.model.init_hidden(self.model.ldims) for _ in range(4)]
                if iSentence % 100 == 0 and iSentence != 0:
                    print('Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Labeled Errors:', (float(lerrors) / etotal) , 'Time', time.time()-start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    lerrors = 0
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                conll_sentence = conll_sentence[1:] + [conll_sentence[0]]
                dloss, deerrors, dlerrors, detotal = self.model.train(conll_sentence, errs)
                eloss += dloss
                mloss += dloss
                eerrors += deerrors
                lerrors += dlerrors
                etotal += detotal
                if len(errs) > 0: # or True:
                    eerrs = torch.sum(cat(errs))
                    eerrs.backward()
                    self.trainer.step()
                    errs = []
                    self.trainer.zero_grad()
                    self.model.init()
        if len(errs) > 0:
            eerrs = torch.sum(cat(errs)) # * (1.0/(float(len(errs))))
            eerrs.backward()
            self.trainer.step()
            errs = []
            self.trainer.zero_grad()
        self.trainer.step()
        print("Loss: ", mloss/iSentence)
