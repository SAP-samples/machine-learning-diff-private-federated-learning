# Differentially Private Federated Learning: A client-level perspective

## Description:
Federated Learning is a privacy preserving decentralized learning protocol introduced by Google. Multiple clients jointly learn a model without data centralization. Centralization is pushed from data space to parameter space: https://research.google.com/pubs/pub44822.html [1].
Differential privacy in deep learning is concerned with preserving privacy of individual data points: https://arxiv.org/abs/1607.00133 [2].
In this work we combine the notion of both by making federated learning differentially private. We focus on preserving privacy for the entire data set of a client. For more information, please refer to: https://arxiv.org/abs/1712.07557v2.

This code simulates a federated setting and enables federated learning with differential privacy. The privacy accountant used is from https://arxiv.org/abs/1607.00133 [2]. The files: accountant.py, utils.py, gaussian_moments.py are taken from: https://github.com/tensorflow/models/tree/master/research/differential_privacy

Note that the privacy agent is not completely set up yet (especially for more than 100 clients). It has to be specified manually or otherwise parameters 'm' and 'sigma' need to be specified.

## Requirements
- [Tensorflow 1.4.1](https://www.tensorflow.org/)
- [MNIST data-set](http://yann.lecun.com/exdb/mnist/)

## Download and Installation
1. Install Tensorflow 1.4.1
2  [Download the files as a ZIP archive](archive/master.zip), or you can [clone the repository](https://help.github.com/articles/cloning-a-repository/) to your local hard drive.

3. Change to the directory of the download, If using macOS, simply run: 
    ```bash
    bash RUNME.sh
    ```
    This will download the [MNIST data-sets](http://yann.lecun.com/exdb/mnist/), create clients and getting started. 
    
For more information on the individual functions, please refer to their doc strings.  

## Known Issues
No issues known


## How to obtain support
This project is provided "as-is" and any bug reports are not guaranteed to be fixed.


## Citations
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

## References
- H. Brendan McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data, 2017, http://arxiv.org/abs/1602.05629.

- Martin Abadi et al., Deep Learning with Differential Privacy, 2016, https://arxiv.org/abs/1607.00133.


## License

This project is licensed under SAP Sample Code License Agreement except as noted otherwise in the [LICENSE file](LICENSE.md).
