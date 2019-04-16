# autodp: Automating differential privacy computation

### Author: 
Yu-Xiang Wang

### Highlights:

1. An RDP (Renyi Differential Privacy) based analytical Moment Accountant implementation that is numerically stable.
2. Supports privacy amplification for generic RDP algorithm for subsampling without replacement and poisson sampling.
3. A privacy calibrator that numerically calibrates noise to privacy requirements using RDP.
4. Bring Your Own Mechanism:  Just implement the RDP of your own DP algorithm as a function.

### How to use?
```
pip install autodp .
```
Then follow the Jupyter notebooks in the `tutorials` folder to get started.


Currently we support only Python3. 


### Research Papers:

  * Wang, Yu-Xiang, Borja Balle, and Shiva Kasiviswanathan. (2019) ["Subsampled Renyi Differential Privacy and Analytical Moments Accountant."](https://arxiv.org/abs/1808.00087). in AISTATS-2019  (**Notable Paper Award**).
