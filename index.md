---
layout: post
title: Contents
---
<span class="newthought">These notes</span> form a concise introductory course on probabilistic graphical models{% include sidenote.html id="note-pgm" note="Probabilistic graphical models are a subfield of machine learning that studies how to describe and reason about the world in terms of probabilities." %}.
They are based on Stanford [CS228](https://cs228.stanford.edu/), and are written by [Volodymyr Kuleshov](http://www.stanford.edu/~kuleshov) and [Stefano Ermon](http://cs.stanford.edu/~ermon/), with the [help](https://github.com/ermongroup/cs228-notes/commits/master) of many students and course staff.
{% include marginnote.html id='mn-construction' note='The notes are still **under construction**! Although we have written up most of the material, you will probably find several typos. If you do, please let us know, or submit a pull request with your fixes to our [GitHub repository](https://github.com/ermongroup/cs228-notes).'%}
You too may help make these notes better by submitting your improvements to us via [GitHub](https://github.com/ermongroup/cs228-notes).

This course starts by introducing probabilistic graphical models from the very basics and concludes by explaining from first principles the [variational auto-encoder](extras/vae), an important probabilistic model that is also one of the most influential recent results in deep learning.

## Sports Analytics

1. [Introduction](analytics/positionless/): What is probabilistic graphical modeling? Overview of the course.

2. [Review of probability theory](preliminaries/probabilityreview): Probability distributions. Conditional probability. Random variables (*under construction*).

3. [Examples of real-world applications](preliminaries/applications): Image denoising. RNA structure prediction. Syntactic analysis of sentences. Optical character recognition (*under construction*).

## Graphics

1. [Bayesian networks](representation/directed/): Definitions. Representations via directed graphs. Independencies in directed models.

2. [Markov random fields](representation/undirected/): Undirected vs directed models. Independencies in undirected models. Conditional random fields.

## Algos From Scratch

1. [Variable elimination](inference/ve/) The inference problem. Variable elimination. Complexity of inference.

2. [Belief propagation](inference/jt/): The junction tree algorithm. Exact inference in arbitrary graphs. Loopy Belief Propagation.

3. [MAP inference](inference/map/): Max-sum message passing. Graphcuts. Linear programming relaxations. Dual decomposition.

4. [Sampling-based inference](inference/sampling/): Monte-Carlo sampling. Forward Sampling. Rejection Sampling. Importance sampling. Markov Chain Monte-Carlo. Applications in inference.

5. [Variational inference](inference/variational/): Variational lower bounds. Mean Field. Marginal polytope and its relaxations.
