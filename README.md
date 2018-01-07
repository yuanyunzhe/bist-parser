# BIST Parsers
## A hybird of a Graph-based and a Transition-based parser using BiLSTM

This project is a hybird of a graph-based and a transition-based parser. The main principle comes from the paper [Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations](https://www.transacl.org/ojs/index.php/tacl/article/viewFile/885/198).

Our repository is forked from elikip's [repo](https://github.com/elikip/bist-parser). He wrote the code with DyNet. We use Python 3.6 with PyTorch, and my cooperator [xiezhq-hermann](https://github.com/xiezhq-hermann) rewrites the [graph-based parser](https://github.com/xiezhq-hermann/bist-parser) using Python3.


#### Required software

 * Python 3.6
 * PyTorch >= 0.2.0
 * (https://github.com/clab/dynet/tree/master/python)


#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
