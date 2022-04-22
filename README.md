# localCHL
This repository implements a version of J. Li et al's Lifted Proximal Operator Machine ([LPOM](https://ojs.aaai.org/index.php/AAAI/article/download/4323/4201)) training of fully connected neural networks. The variant of LPOM implemented here is described by C. Zach in [Bilevel Programs Meet Deep Learning](https://arxiv.org/abs/2105.07231).
LPOM belongs to a family of training algorithms (Lifted Networks, Contrastive Hebbian Learning, Equilibrium Propagation etc.), which do not use backpropagation, but instead transports error signals during activation inference.
The tradeoff is that activation inference has no closed form solution and must be solved iteratively (requiring 10-100 iterations), which typically makes these models much slower than backproop.
By using LoopVectorization.jl this implementation manages to stay fairly close to the performance of BP for smaller networks (2-3 layers and ~100 neurons per layer), inspite of solving each individual neurons state via coordinate descent 10 times per datapoint per epoch.

This is one of the first julia projects I implemented, so it doesn't quite utilize the Julia ecosystem as well as it could have done. At some point I should rewrite it to make use of more functionality from Flux.jl.
