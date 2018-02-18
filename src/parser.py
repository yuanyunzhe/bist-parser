# -*- coding：utf-8 -*-
from optparse import OptionParser
import pickle, utils, os, time, sys, multiprocessing
import torch
from hybird import Hybrid

if __name__ == '__main__':
    parser = OptionParser()

    # Default parser
    parser.add_option("--parser", type="string", dest="Trinsition-based or Graph-based or Hybird parser", default="hybird")

    # GPU and multiprocessing
    parser.add_option("--gpu", type="int", dest="gpu", default=0)
    parser.add_option("--gpu2", type="int", dest="gpu", default=1)
    parser.add_option("--batch", type="int", dest="batch", default=1)
    parser.add_option("--numthread", type="int", dest="numthread", default=8)

    # I/O
    parser.add_option("--outdir", type="string", dest="output", default="results")
    # parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default="corpus/taobao1/train-drop.conll")
    # parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default="corpus/taobao1/dev-drop.conll")
    # parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default="corpus/taobao1/test-drop.conll")
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default="corpus/ud/en-ud-train-dropped.conllu")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default="corpus/ud/en-ud-dev-dropped.conllu")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default="corpus/ud/en-ud-test-dropped.conllu")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="barchybrid.model")
    parser.add_option("--multi", dest="multi", help="Annotated CONLL multi-train file", metavar="FILE", default=False)
                      # multi-task has been deleted for bloated code
    # Embedding
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=25)
    parser.add_option("--rembedding", type="int", dest="rembedding_dims", default=25)
    parser.add_option("--oembedding", type="int", dest="oembedding_dims", default=0) #ontology
    parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=25) #cpos

    # Learning
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--optim", type="string", dest="optim", default='adam')
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--k", type="int", dest="window", default=3)
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=50)
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=125)

    # Flags only for transition-based
    parser.add_option("--disableoracle", action="store_false", dest="oracle", default=True)
    parser.add_option("--usehead", action="store_true", dest="headFlag", default=False)
    parser.add_option("--userlmost", action="store_true", dest="rlFlag", default=False)
    parser.add_option("--userl", action="store_true", dest="rlMostFlag", default=False)

    # Flag for training or predict
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)

    (options, args) = parser.parse_args()

    if torch.cuda.is_available() and options.gpu >= 0:
        torch.cuda.set_device(options.gpu)
        print("Using GPU")
    else:
        torch.cuda.is_available=lambda:False
        print("Using CPU")

    max_thread = multiprocessing.cpu_count()
    active_thread = options.numthread if max_thread > options.numthread else max_thread
    torch.set_num_threads(active_thread)
    print(active_thread, "threads are in use")

    print('Using external embedding:', options.external_embedding)

    if not options.predictFlag:
        if not (options.rlFlag or options.rlMostFlag or options.headFlag):
            print('You must use either --userlmost or --userl or --usehead (you can use multiple)')
            sys.exit()

        print('Preparing vocab')
        words, enum_word, pos, rels, onto, cpos = list(utils.vocab(options.conll_train))
        with open(os.path.join(options.output, options.params), 'wb') as paramsfp:
            pickle.dump((words, enum_word, pos, rels, onto, cpos, options), paramsfp)
        print('Finished collecting vocab')

        print('Initializing blstm arc hybrid:')
        parser = Hybrid(words, pos, rels, enum_word, options, onto, cpos)

        for epoch in range(options.epochs):
            print('Starting epoch', epoch)
            parser.train(options.conll_train)
            conllu = (os.path.splitext(options.conll_dev.lower())[1] == '.conllu')
            devpath = os.path.join(options.output, 'dev_epoch_' + str(epoch + 1) + ('.conll' if not conllu else '.conllu'))
            utils.write_conll(devpath, parser.predict(options.conll_dev))
            parser.save(os.path.join(options.output, os.path.basename(options.model) + str(epoch + 1)))

            if not conllu:
                os.system('perl src/utils/eval.pl -g ' + options.conll_dev  + ' -s ' + devpath  + ' > ' + devpath + '.txt')
                with open(devpath + '.txt', 'r', encoding='UTF-8') as f:
                    for i in range(0, 3):
                        print(f.readline())
            else:
                os.system('python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + options.conll_dev + ' ' + devpath + ' > ' + devpath + '.txt')
                with open(devpath + '.txt', 'r', encoding='UTF-8') as f:
                    for l in f:
                        if l.startswith('UAS'):
                            print('UAS:%s' % l.strip().split()[-1])
                        elif l.startswith('LAS'):
                            print('LAS:%s' % l.strip().split()[-1])
            print('Finished predicting dev')
    else:
        with open(options.params, 'rb') as paramsfp:
            words, enum_word, pos, rels, onto, cpos, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = options.external_embedding

        print('Initializing lstm mstparser:')
        parser = Hybrid(words, pos, rels, enum_word, stored_opt, onto, cpos)
        parser.load(options.model)
        conllu = (os.path.splitext(options.conll_test.lower())[1] == '.conllu')
        testpath = os.path.join(options.output, 'test_pred.conll' if not conllu else 'test_pred.conllu')
        ts = time.time()
        pred = list(parser.predict(options.conll_test))
        te = time.time()
        print('Finished predicting test',te - ts)
        utils.write_conll(testpath, pred)

        if not conllu:
            os.system('perl src/utils/eval.pl -g ' + options.conll_test + ' -s ' + tespath  + ' > ' + tespath + '.txt')
        else:
            os.system('python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + options.conll_test + ' ' + tespath + ' > ' + testpath + '.txt')
            with open(testpath + '.txt', 'r', encoding='UTF-8') as f:
                for l in f:
                    if l.startswith('UAS'):
                        print('UAS:%s' % l.strip().split()[-1])
                    elif l.startswith('LAS'):
                        print('LAS:%s' % l.strip().split()[-1])
