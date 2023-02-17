# Guided Deep Kernel Learning
Combining Gaussian processes with the expressive power of deep neural networks is commonly done nowadays through deep kernel learning (DKL). Unfortunately, due to the kernel optimization process, this often results in losing their Bayesian benefits. In this study, we present a novel approach for learning deep kernels by utilizing infinite-width neural networks. We propose to use the Neural Network Gaussian Process (NNGP) model as a guide to the DKL model in the optimization process. Our approach harnesses the reliable uncertainty estimation of the NNGPs to adapt the DKL confidence when it encounters novel data points. As a result, we get the best of both worlds, we leverage the Bayesian behavior of the NNGP, namely its robustness to overfitting, and accurate uncertainty estimation, while maintaining the generalization abilities, scalability, and flexibility of deep kernels. Empirically, we show on multiple benchmark datasets of varying sizes and dimensionality, that our method is robust to overfitting, has good predictive performance, and provides reliable uncertainty estimations.


### Instructions
Install repo:
```bash
pip install -e .
```

To run on Buzz and CTSlice, you first need to download the datasets from the repository of
"Semi-supervised Deep Kernel Learning: Regression with Unlabeled Data by Minimizing Predictive Variance"
at the following [[link]](https://github.com/ermongroup/ssdkl)

To run GDKL first got the required directory (either uci or cifar):
```bash
cd experiments/dataset
```
and then run trainer_GDKL.py or trainer_GDKL_sparse.py

