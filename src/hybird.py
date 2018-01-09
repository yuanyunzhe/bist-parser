# -*- coding：utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from utils import ParseForest, read_conll, write_conll
from extend import *

import utils, time, random
import shutil

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

    def predict(self, conll_path):
        self.transitionModel.init()
        num_g, num_t = 0, 0
        with open(conll_path, "r") as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP,False)):
                self.graphModel.hid_for_1, self.graphModel.hid_back_1, self.graphModel.hid_for_2, self.graphModel.hid_back_2 = [
                    self.graphModel.init_hidden(self.graphModel.ldims) for _ in range(4)]
                conll_sentence0 = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                conll_sentence1 = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                conll_sentence = [conll_sentence0, conll_sentence1]

                self.graphModel.predict(conll_sentence0)
                self.transitionModel.hid_for_1, self.transitionModel.hid_back_1, self.transitionModel.hid_for_2, self.transitionModel.hid_back_2 = [
                    self.transitionModel.init_hidden(self.transitionModel.ldims) for _ in range(4)]
                conll_sentence1 = conll_sentence1[1:] + [conll_sentence1[0]]
                self.transitionModel.predict(conll_sentence1)

                input = torch.cat((self.graphModel.vec, self.transitionModel.vec), 1)
                output = self.classifier(Variable(input))

                _, rank = torch.max(torch.abs(output.data), 1)
                rank = rank[0]
                if rank == 0:
                    num_g += 1
                else:
                    num_t += 1

                # rank = random.randint(0, 1)
                yield conll_sentence[rank]
        print("Graph-based:", num_g, "\nTransition-based:", num_t)

    def train(self, conll_path):

        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP, True))
            random.shuffle(shuffledData)
            errs_g = []
            errs_t = []
            lerrs = []
            self.transitionModel.init()

            parse_vec = []
            parse_lbl = []

            for iSentence, sentence in enumerate(shuffledData):
                self.graphModel.hid_for_1, self.graphModel.hid_back_1, self.graphModel.hid_for_2, self.graphModel.hid_back_2 = [
                    self.graphModel.init_hidden(self.graphModel.ldims) for _ in range(4)]
                self.transitionModel.hid_for_1, self.transitionModel.hid_back_1, self.transitionModel.hid_for_2, self.transitionModel.hid_back_2 = [
                    self.transitionModel.init_hidden(self.transitionModel.ldims) for _ in range(4)]

                if iSentence % 100 == 0 and iSentence != 0:
                    print('Processing sentence number:', iSentence,
                          'Loss:', eloss / etotal,
                          'Errors:', (float(eerrors)) / etotal,
                          'Time', time.time() - start)
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0

                conll_sentence0 = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                graphLoss = self.graphModel.forward(conll_sentence0, errs_g, lerrs)
                eerrors += graphLoss
                eloss += graphLoss
                mloss += graphLoss
                etotal += len(sentence)

                conll_sentence1 = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                conll_sentence1 = conll_sentence1[1:] + [conll_sentence1[0]]
                transitionLoss, deerrors, dlerrors, detotal = self.transitionModel.forward(conll_sentence1, errs_t)
                eloss += transitionLoss
                mloss += transitionLoss
                eerrors += deerrors
                etotal += detotal

                parse_vec.append(torch.cat((self.graphModel.vec, self.transitionModel.vec), 1))
                parse_lbl.append(0 if graphLoss * 3 < transitionLoss else 1)

                # print((self.graphModel.vec - self.transitionModel.vec) / self.graphModel.vec)

                if len(errs_g) > 0 or len(lerrs) > 0:
                    eerrs_g = torch.sum(cat(errs_g + lerrs))
                    eerrs_g.backward()
                if len(errs_t) > 0:
                    eerrs_t = torch.sum(cat(errs_t))
                    eerrs_t.backward()

                if len(errs_g) > 0 or len(lerrs) > 0:
                    self.graphTrainer.step()
                    errs_g = []
                    lerrs = []
                if len(errs_t) > 0:
                    self.transitionTrainer.step()
                    errs_t = []

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
            output = self.classifier(input)
            output = self.loss(output, label)
            self.optimizer.zero_grad()
            output.backward()
            self.optimizer.step()


class Hybrid:
    def __init__(self, vocab, pos, rels, enum_word, options, onto, cpos):
        model = HybridModel(vocab, pos, rels, enum_word, options, onto, cpos)
        self.model = model.cuda() if torch.cuda.is_available() else model

    def predict(self, conll_path):
        return self.model.predict(conll_path)

    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.model.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.model.load_state_dict(torch.load(fn))

    def train(self, conll_path):
        self.model.train(conll_path)


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
