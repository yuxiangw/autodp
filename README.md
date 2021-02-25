# autodp: Automating differential privacy computation 

### Highlights

* Advanced DP techniques (e.g., Renyi DP, Moments Accountant, f-DP) working behind the scene.
* Easily customizable.  Bring your own mechanism in any way you want.
* Strong composition over heterogeneous mechanisms.

### All new autodp "Mechanism" API
<img src="https://github.com/yuxiangw/autodp/blob/master/figures/autodp_design.png" alt="The three main classes of the autodp 'mechanism' API." width="800x" height="208x"/>

### New features that come with the new API

1. Object oriented design: see   check out ```autodp_core.py```
2. Zoos are open with many private animals:  ```mechanism_zoo```,  ```transformer_zoo```, ```calibrator_zoo```. 
3. Added support for f-DP and privacy profile alongside RDP.
4. Stronger RDP to (eps,delta)-DP conversion.
5. Privacy amplification by X.
6. Exactly tight privacy accounting for Gaussian mechanisms and their compositions.
7. Interpretable privacy guarantee via Hypothesis testing interpretation for any Mechanism.


The new API makes it extremely easy to obtain state-of-the-art privacy guarantees for your favorite randomized mechanisms, with just a few lines of codes.


### How to use?

It's easy. Just run:
```
pip install autodp
```
or

```
pip3 install autodp
``` 
Check out the Jupyter notebooks in the `tutorials` folder to get started.

#### Notes:
* ```pip``` should automatically install all the dependences for you.
* Currently we support only Python3.
* You might need to run ```pip install autodp --upgrade```


###  To use the current version at the master branch

Install it locally by:
```
pip install -e .
```

### Research Papers:

  * Yu-Xiang Wang, Borja Balle, and Shiva Kasiviswanathan. (2019) ["Subsampled Renyi Differential Privacy and Analytical Moments Accountant."](https://arxiv.org/abs/1808.00087). in AISTATS-2019  (**Notable Paper Award**).
  * Yuqing Zhu, Yu-Xiang Wang. (2019) ["Poisson Subsampled Renyi Differential Privacy"](http://proceedings.mlr.press/v97/zhu19c.html). ICML-2019.
  * Yuqing Zhu, Yu-Xiang Wang. (2020) ["Improving Sparse Vector Technique with Renyi Differential Privacy"](https://papers.nips.cc/paper/2020/hash/e9bf14a419d77534105016f5ec122d62-Abstract.html). in NeurIPS-2020.


### How to Contribute?

Follow the standard practice. Fork the repo, create a branch, develop the edit and send a pull request. One of the maintainers are going to review the code and merge the PR. Alternatively, please feel free to creat issues to report bugs, provide comments and suggest new features. 

At the moment, contributions to examples, tutorials, as well as the RDP of currently unsupported mechanisms are most welcome (add them to ```RDP_bank.py```)! 
Also, you may add new mechanisms to ``mechanism_zoo.py``. Contributions to ``transformer_zoo.py`` and ``calibrator_zoo.py`` are trickier, please email us! 

Please explain clearly what the contribution is about in the PR and attach/cite papers whenever appropriate.


### Legacy: the moments accountant API from autodp v.0.11 is still supported: 

1. An RDP (Renyi Differential Privacy) based analytical Moment Accountant implementation that is numerically stable.
2. Supports privacy amplification for generic RDP algorithm for subsampling without replacement and poisson sampling.
3. Stronger composition than the optimal composition using only (&epsilon;,&delta;)-DP.
4. A privacy calibrator that numerically calibrates noise to privacy requirements using RDP.
5. Bring Your Own Mechanism:  Just implement the RDP of your own DP algorithm as a function.



### Examples：

<img src="https://github.com/yuxiangw/autodp/blob/master/figures/gaussian_compose_mean.png" alt="Composing Subsampled Gaussian Mechanisms (high noise)" width="400x"/><img src="https://github.com/yuxiangw/autodp/blob/master/figures/LN_gaussian_compose_mean.png" alt="Composing Subsampled Gaussian Mechanisms (low noise)" width="400x"/>

**Figure 1**: Composing subsampled Gaussian Mechanisms. *Left*: High noise setting with &sigma;=5, &gamma;=0.001, &delta;=1e-8.  *Right*: Low noise setting with &sigma;=0.5, &gamma;=0.001, &delta;=1e-8.


<img src="https://github.com/yuxiangw/autodp/blob/master/figures/laplace_compose_mean.png" alt="Composing Subsampled Laplace Mechanisms (high noise)" width="400x"/><img src="https://github.com/yuxiangw/autodp/blob/master/figures/LN_laplace_compose_mean.png" alt="Composing Subsampled Laplace Mechanisms (low noise)" width="400x"/>

**Figure 2**: Composing subsampled Laplace Mechanisms. *Left*: High noise setting with b=2, &gamma;=0.001, &delta;=1e-8.  *Right*: Low noise setting with b=0.5, &gamma;=0.001, &delta;=1e-8.
