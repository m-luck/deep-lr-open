#
deep-lr

### Project Abstract:

We are using neural networks to facilitate lip reading trained on the [GRID Corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/) (an alternative to the BBC's LRW dataset). We will augment the [LipNet architecture](https://arxiv.org/pdf/1611.01599.pdf) by incorporating a language model such as GPT-2. Because the sentence structure of the GRID Corpus is structured/constrained, we will have to fine tune the language model on the GRID Corpus' training text. We hope to demonstrate superior results in comparison to the original LipNet architecture. If this is successful and we eventually obtain access to the LRW dataset, we could try augmenting the original [WLAS model](https://arxiv.org/pdf/1611.05358.pdf) in a similar fashion.   

### TODO Timeline:

1) Verify Lipnet, port to PyTorch
[DONE] 2) Get GPT-2 on PyTorch
3) Cat GPT-2 and Lipnet 

LipNet in PyTorch for dev reference: https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66 
