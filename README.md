# autodp: Automating differential privacy computation

### Author: 
Yu-Xiang Wang

### Highlights:

1. An RDP (Renyi Differential Privacy) based analytical Moment Accountant implementation that is numerically stable.
2. Supports privacy amplification for generic RDP algorithm for subsampling without replacement and poisson sampling.
3. Stronger composition than the optimal composition using only (&epsilon;,&delta;)-DP.
4. A privacy calibrator that numerically calibrates noise to privacy requirements using RDP.
5. Bring Your Own Mechanism:  Just implement the RDP of your own DP algorithm as a function.

### How to use?

It's easy. Just run:
```
pip install autodp
```
Then follow the Jupyter notebooks in the `tutorials` folder to get started.

#### Notes:
* ```pip``` should automatically install all the dependences for you.
* Currently we support only Python3. 
* You might need to run ```pip3 install autodp --upgrade```

### Research Papers:

  * Yu-Xiang Wang, Borja Balle, and Shiva Kasiviswanathan. (2019) ["Subsampled Renyi Differential Privacy and Analytical Moments Accountant."](https://arxiv.org/abs/1808.00087). in AISTATS-2019  (**Notable Paper Award**).


### Examples of composing over many rounds


<object data="https://github.com/yuxiangw/autodp/blob/master/figures/gaussian_compose_mean.pdf" type="application/pdf" width="400px" height="400px">
    <embed src="https://github.com/yuxiangw/autodp/blob/master/figures/gaussian_compose_mean.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://yoursite.com/the.pdf">Download PDF</a>.</p>
    </embed>
</object><object data="https://github.com/yuxiangw/autodp/blob/master/figures/LN_gaussian_compose_mean.pdf" type="application/pdf" width="400px" height="400px">
    <embed src="https://github.com/yuxiangw/autodp/blob/master/figures/LN_gaussian_compose_mean.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://yoursite.com/the.pdf">Download PDF</a>.</p>
    </embed>
</object>

**Figure 1**: Composing subsampled Gaussian Mechanisms. Left: High noise setting with &sigma;=5, &gamma;=0.001, &delta;=1e-8.  Right: Low noise setting with &sigma;=0.5, &gamma;=0.001, &delta;=1e-8.



<object data="https://github.com/yuxiangw/autodp/blob/master/figures/laplace_compose_mean.pdf" type="application/pdf" width="400px" height="400px">
    <embed src="https://github.com/yuxiangw/autodp/blob/master/figures/laplace_compose_mean.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://yoursite.com/the.pdf">Download PDF</a>.</p>
    </embed>
</object> <object data="https://github.com/yuxiangw/autodp/blob/master/figures/LN_laplace_compose_mean.pdf" type="application/pdf" width="400px" height="400px">
    <embed src="https://github.com/yuxiangw/autodp/blob/master/figures/LN_laplace_compose_mean.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://yoursite.com/the.pdf">Download PDF</a>.</p>
    </embed>
</object>

**Figure 1**: Composing subsampled Laplace Mechanisms. Left: High noise setting with b=2, &gamma;=0.001, &delta;=1e-8.  Right: Low noise setting with b=0.5, &gamma;=0.001, &delta;=1e-8.
