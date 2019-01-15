Master Thesis: Bayesian Convolutional Neural Networks 
======================================================

> Thesis work submitted at Computer Science department at University of Kaiserslautern.

[![License MIT](http://img.shields.io/badge/license-MIT-brightgreen.svg)](license.md)

## Author
*   [Kumar Shridhar](https://kumar-shridhar.github.io/)

## Supervisors
*   Prof. Marcus Liwicki (Professor at Luleå Unoversity, Sweden)
*   Felix Laumann (PhD candidate at Imperial College, London)


---------------------------------------------------------------------------------------------------------

## Abstract

Artificial Neural Networks are connectionist systems that perform a given task by learning on examples without having a prior knowledge about the task. This is done by finding an optimal point estimate for the weights in every node.
Generally, the network using point estimates as weights perform well with large datasets, but they fail to express uncertainty in regions with little or no data, leading to overconfident decisions.

In this thesis, **Bayesian Convolutional Neural Network (BayesCNN) using Variational Inference** is proposed, that introduces probability distribution over the weights. Furthermore, the proposed BayesCNN architecture is applied to tasks like Image Classification, Image Super-Resolution and Generative Adversarial Networks.

BayesCNN is based on **Bayes by Backprop** which derives a variational approximation to the true posterior. 
Our proposed method not only achieves performances equivalent to frequentist inference in identical architectures but also incorporate a measurement for uncertainties and regularisation. It further eliminates the use of dropout in the model. Moreover, we predict how certain the model prediction is based on the epistemic and aleatoric uncertainties and finally, we propose ways to prune the Bayesian architecture and to make it more computational and time effective. 

In the first part of the thesis, the Bayesian Neural Network is explained and it is applied to an Image Classification task. The results are compared to point-estimates based architectures on MNIST, CIFAR-10, CIFAR-100 and STL-10 datasets. Moreover, uncertainties are calculated and the architecture is pruned and a comparison between the results is drawn.

In the second part of the thesis, the concept is further applied to other computer vision tasks namely, Image Super-Resolution and Generative Adversarial Networks. The concept of BayesCNN is tested and compared against other concepts in a similar domain.

---------------------------------------------------------------------------------------------------------

## Code base

The proposed work has been implemented in PyTorch and is available here : [BayesianCNN](https://github.com/kumar-shridhar/PyTorch-BayesianCNN)

---------------------------------------------------------------------------------------------------------

## Chapter Overview


### Chapter 1 : Introduction

*   Why there is a need for Bayesian Networks?

*   Problem Statement 

*   Current Situation

*   Our Hypothesis

*   Our Contribution


### Chapter 2: Background

*   Neural Networks and Convolutional Neural Networks

*   Concepts overview of Variational Inference, and local reparameterization trick in Bayesian Neural Network.

*   Backpropagation in Bayesian Networks using Bayes by Backprop.

*   Estimation of Uncertainties in a network.

*   Pruning a network to reduce the number of overall parameters without affecting it's performance.


### Chapter 3: Related Work

*   How Bayesian Methods were applied to Neural Networks for the intractable true posterior distribution.

*   Various ways of training Neural Networks posterior probability distributions: Laplace approximations, Monte Carlo and    Variational Inference.

*   Proposals on Dropout and Gaussian Dropout as Variational Inference schemes.

*   Work done in the past for uncertainty estimation in Neural Network.

*   Ways to reduce the number of parameters in a model.

### Chapter 4: Concept

*   Bayesian CNN with Variational Inference based on Bayes by Backprop.

*   Bayesian convolutional operations with mean and variance.

*   Local reparameterization trick for Bayesian CNN.

*   Uncertainty estimation in a Bayesian network.

*   Using L1 norm for reducing the number of parameters in a Bayesian network.

### Chapter 5: Empirical Analysis

*   Applying Bayesian CNN for the task of Image Recognition on MNIST, CIFAR-10, CIFAR-100 and STL-10 datasets.

*   Comparison of results of Bayesian CNN with Normal CNN architectures on similar datasets.

*   Regularization effect of Bayesian Network with dropouts.

*   Distribution of mean and variance in Bayesian CNN over time.

*   Parameters comparison before and after model pruning. 

### Chapter 6: Applications

*   Empirical analysis of BayesCNN with normal architecture for Image Super Resolution.

*   Empirical analysis of BayesCNN with normal architecture for Generative Adversarial Networks.

### Chapter 7: Conclusion and Outlook

*   Conclusion

### Appendix A

*   Experiment Specification

### Appendix B

*   How to replicate results

----------------------------------------------------------------------------------------------------------


### Paper

*  Journal paper of this work is also available on Arxiv: [
A Comprehensive guide to Bayesian Convolutional Neural Network with Variational Inference](https://arxiv.org/abs/1901.02731)

*   Feel free to cite, if the work is of any help to you:

```
@article{shridhar2019comprehensive,
  title={A Comprehensive guide to Bayesian Convolutional Neural Network with Variational Inference},
  author={Shridhar, Kumar and Laumann, Felix and Liwicki, Marcus},
  journal={arXiv preprint arXiv:1901.02731},
  year={2019}
}
```
----------------------------------------------------------------------------------------------------------

## Thesis Template

*   Cambridge Computer Laboratory PhD Thesis Template [https://github.com/cambridge/thesis](https://github.com/cambridge/thesis)

----------------------------------------------------------------------------------------------------------

### Contact

*   shridhar.stark@gmail.com

---------------------------------------------------------------------------------------------------------

