# Differentially Private Federated Learning: A client-level perspective

Introduction:
============
Federated Learning is a privacy preserving decentralized learning protocol introduced my Google. Multiple clients jointly learn a model without data centralization. Centralization is pushed from data space to parameter space: https://research.google.com/pubs/pub44822.html [1].
Differential privacy in deep learning is concerned with preserving privacy of individual data points: https://arxiv.org/abs/1607.00133 [2].
In this work we combine the notion of both by making federated learning differentially private. We focus on preserving privacy for the entire data set of a client. For more information, please refer to: https://arxiv.org/abs/1712.07557v2.

This code simulates a federated setting and enables federated learning with differential privacy. The privacy accountant used is from https://arxiv.org/abs/1607.00133 [2]. The files: accountant.py, utils.py, gaussian_moments.py are taken from: https://github.com/tensorflow/models/tree/master/research/differential_privacy

Installation
============
1- If not already done, install Tensorflow

2- Downlad the directory this README is part of 

3- If using macOS, simply run: 
```bash
bash RUNME.sh
```
to download the MNIST data-sets, create clients and getting started. For more information on the individual functions, please refer to their doc strings.  

How to Cite
===========
If you use this code or the pretrained models in your research,
please cite:

```
@ARTICLE{2017arXiv171207557G,
   author = {{Geyer}, R.~C. and {Klein}, T. and {Nabi}, M.},
    title = "{Differentially Private Federated Learning: A Client Level Perspective}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1712.07557},
 primaryClass = "cs.CR",
 keywords = {Computer Science - Cryptography and Security, Computer Science - Learning, Statistics - Machine Learning},
     year = 2017,
    month = dec,
   adsurl = {http://adsabs.harvard.edu/abs/2017arXiv171207557G},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

Reference(s)
===========
[1] H. Brendan McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data, 2017, http://arxiv.org/abs/1602.05629.

[2] Martin Abadi et al., Deep Learning with Differential Privacy, 2016, https://arxiv.org/abs/1607.00133.
