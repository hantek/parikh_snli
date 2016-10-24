# A Lasagne Implementation of Parikh's Decomposable Attention Model

This repo tries to reproduce Parikh's decomposable attention model described in this paper:

[A Decomposable Attention Model for Natural Language Inference(https://arxiv.org/abs/1606.01933v1)](https://arxiv.org/abs/1606.01933v1)

To run the code, you will need Theano and Lasagne to be installed in your
computer. You will also need the trained GloVe word embedding (we are using the
[glove.840B.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip) version),
and definitely the [SNLI
dataset](http://nlp.stanford.edu/projects/snli/snli_1.0.zip)

The `oov_vec.py` file is used to preprocess the data into a `.pkl` format.
The main script is `parikh.py`, run it with proper command line args. You can
find the suggested args in the file.


