# AMTNN
Adversarial Multitask Neural Network 

A pytorch implementation of [A Principled Approach for Learning Task Similarity in Multitask Learning](https://arxiv.org/abs/1903.09109)


## Prerequisites

- Pytorch >=1.0, Torchvision >=0.2 
- CVXPY = 1.0

## Models

- 'MTL.py': MTL model for digit classification problem (with H divergence and Wasserstein distance options)
- 'alpha_opt.py': Model for solving the convex optimization problem (Eq (2) in the paper)

## How to cite

```xml
@inproceedings{ijcai2019-478,
  title     = {A Principled Approach for Learning Task Similarity in Multitask Learning},
  author    = {Shui, Changjian and Abbasi, Mahdieh and Robitaille, Louis-Émile and Wang, Boyu and Gagné, Christian},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI-19}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {3446--3452},
  year      = {2019},
  month     = {7},
  doi       = {10.24963/ijcai.2019/478},
  url       = {https://doi.org/10.24963/ijcai.2019/478},
}
```

