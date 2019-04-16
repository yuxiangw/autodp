# autodp: A package to automate differential privacy computation

### Author: 
Yu-Xiang Wang

### Highlights:

1. An RDP based analytical Moment Accountant implementation that is numerically stable.
2. Supports privacy amplification for generic RDP algorithm for subsampling without replacement and poisson sampling.
3. A privacy calibrator that numerically calibrates noise to privacy requirements using RDP.

### How to use?
```
pip install autodp .
```
Then follow the Jupyter notebooks in the `tutorials` folder to get started.


Currently we support only Python3. 


### Research Papers:

..* Wang, Yu-Xiang, Borja Balle, and Shiva Kasiviswanathan. (2019) "Subsampled R\'enyi Differential Privacy and Analytical Moments Accountant." in AISTATS-2019  ($Notable Paper Award$).
