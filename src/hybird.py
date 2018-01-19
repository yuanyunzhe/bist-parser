# -*- coding：utf-8 -*-
import torch
import torch.nn as nn

from torch import optim
from utils import read_conll
from extend import *

import utils, time, random

from graph import GraphModel
from transition import TransitionModel

class HybridModel(nn.Module):
    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos):
        super(HybridModel, self).__init__()
        random.seed(2)
        dims = options.wembedding_dims + options.pembedding_dims + options.cembedding_dims + options.oembedding_dims
        self.share_for = nn.LSTM(dims, options.lstm_dims)
        self.share_back = nn.LSTM(dims, options.lstm_dims)
        model0 = GraphModel(vocab, pos, rels, enum_word, options, onto, cpos, self.share_for, self.share_back)
        model1 = TransitionModel(vocab, pos, rels, enum_word, options, onto, cpos, self.share_for, self.share_back)
        self.graphModel = model0.cuda() if torch.cuda.is_available() else model0
        self.transitionModel = model1.cuda() if torch.cuda.is_available() else model1

        self.graphTrainer = get_optim(options.optim, self.graphModel.parameters())
        self.transitionTrainer = get_optim(options.optim, self.transitionModel.parameters())

        classifier = LinearClassifier(options.lstm_dims * 4)
        self.classifier = classifier.cuda() if torch.cuda.is_available() else classifier
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(classifier.parameters(), lr=0.01)

    def forward(self, sentence):
        pass


class Hybrid(DependencyParser):
    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos):
        model = HybridModel(vocab, pos, rels, enum_word, options, onto, cpos)
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.graphModel = self.model.graphModel
        self.transitionModel = self.model.transitionModel
        self.graphTrainer = self.model.graphTrainer
        self.transitionTrainer = self.model.transitionTrainer

    def predict(self, conll_path):

        self.transitionModel.init()
        num_g, num_t = 0, 0
        with open(conll_path, "r", encoding='UTF-8') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP,False)):
                self.graphModel.hid_for_1, self.graphModel.hid_back_1, self.graphModel.hid_for_2, self.graphModel.hid_back_2 = [
                    self.graphModel.init_hidden(self.graphModel.ldims) for _ in range(4)]
                self.transitionModel.hid_for_1, self.transitionModel.hid_back_1, self.transitionModel.hid_for_2, self.transitionModel.hid_back_2 = [
                    self.transitionModel.init_hidden(self.transitionModel.ldims) for _ in range(4)]
                
                conll_sentence0 = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                conll_sentence1 = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                conll_sentence1 = conll_sentence1[1:] + [conll_sentence1[0]]
                
                self.graphModel.predict(conll_sentence0)
                self.transitionModel.predict(conll_sentence1)

                conll_sentence1 = [conll_sentence1[-1]] + conll_sentence1[:-1]
                conll_sentence = [conll_sentence0, conll_sentence1]
                # rank = random.randint(0, 1)
                input = torch.cat((self.graphModel.vec, self.transitionModel.vec), 1)
                output = self.model.classifier(Variable(input))
                _, rank = torch.max(torch.abs(output.data), 1)
                rank = rank[0]
                num_g += 1 - rank
                num_t += rank
                yield conll_sentence[rank]
        print("Graph-based:", num_g, "\nTransition-based:", num_t)


    def train(self, conll_path):
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        gfor, gback, tfor, tback = 0, 0, 0, 0
        with open(conll_path, 'r', encoding='UTF-8') as conllFP:
            shuffledData = list(read_conll(conllFP, True))
            random.shuffle(shuffledData)
            errs_g, errs_t, lerrs = [], [], []
            parse_vec, parse_lbl = [], []
            self.transitionModel.init()
            for iSentence, sentence in enumerate(shuffledData):
                self.graphModel.hid_for_1, self.graphModel.hid_back_1, self.graphModel.hid_for_2, self.graphModel.hid_back_2 = [
                    self.graphModel.init_hidden(self.graphModel.ldims) for _ in range(4)]
                self.transitionModel.hid_for_1, self.transitionModel.hid_back_1, self.transitionModel.hid_for_2, self.transitionModel.hid_back_2 = [
                    self.transitionModel.init_hidden(self.transitionModel.ldims) for _ in range(4)]
                if iSentence % 100 == 0 and iSentence != 0:
                    print('Processing sentence number:', iSentence,
                          'Loss:', eloss / etotal,
                          'Errors:', (float(eerrors)) / etotal,
                          'Time', time.time() - start,
                          '\nGFor', gfor, 'GBack', gback,
                          '\nTFor', tfor, 'TBack', tback)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    gfor, gback, tfor, tback = 0, 0, 0, 0

                conll_sentence0 = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                conll_sentence1 = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                conll_sentence1 = conll_sentence1[1:] + [conll_sentence1[0]]

                tmp = time.time()
                graphLoss = self.graphModel.forward(conll_sentence0, errs_g, lerrs)
                eerrors += graphLoss
                eloss += graphLoss
                mloss += graphLoss
                etotal += len(sentence)
                gfor += time.time() - tmp

                tmp = time.time()
                transitionLoss, deerrors, detotal = self.transitionModel.forward(conll_sentence1, errs_t)
                eerrors += deerrors
                eloss += transitionLoss
                mloss += transitionLoss
                etotal += detotal
                tfor += time.time() - tmp

                parse_vec.append(torch.cat((self.graphModel.vec, self.transitionModel.vec), 1))
                parse_lbl.append(0 if graphLoss * 3 < transitionLoss else 1)
                # print((self.graphModel.vec - self.transitionModel.vec) / self.graphModel.vec)

                if len(errs_g) > 0 or len(lerrs) > 0:
                    eerrs_g = torch.sum(cat(errs_g + lerrs))
                    tmp = time.time()
                    eerrs_g.backward()
                    gback += time.time() - tmp
                if len(errs_t) > 50:
                    eerrs_t = torch.sum(cat(errs_t))
                    tmp = time.time()
                    eerrs_t.backward()
                    tback += time.time() - tmp

                if len(errs_g) > 0 or len(lerrs) > 0:
                    tmp = time.time()
                    self.graphTrainer.step()
                    errs_g = []
                    lerrs = []
                    gback += time.time() - tmp
                if len(errs_t) > 50:
                    tmp = time.time()
                    self.transitionTrainer.step()
                    errs_t = []
                    tback += time.time() - tmp

                self.graphTrainer.zero_grad()
                self.transitionTrainer.zero_grad()
                self.transitionModel.init()

        if len(errs_g) > 0:
            eerrs_g = (torch.sum(errs_g + lerrs))
            eerrs_g.backward()
        if len(errs_t) > 0:
            eerrs_t = torch.sum(cat(errs_t))
            eerrs_t.backward()

        if len(errs_g) > 0:
            self.graphTrainer.step()
        if len(errs_t) > 0:
            self.transitionTrainer.step()

        self.graphTrainer.zero_grad()
        self.transitionTrainer.zero_grad()
        print("Loss: ", mloss / iSentence)

        input = Variable(torch.cat(parse_vec, 0))
        label = Variable(torch.LongTensor(parse_lbl))

        for i in range(100):
            output = self.model.classifier(input)
            output = self.model.loss(output, label)
            self.model.optimizer.zero_grad()
            output.backward()
            self.model.optimizer.step()


class LinearClassifier(nn.Module):
    def __init__(self, dim):
        super(LinearClassifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 2)
        )

    def forward(self, x):
        return self.main(x)
