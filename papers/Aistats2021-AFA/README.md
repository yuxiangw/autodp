# Overview
This folder contains the implementation of analytical Fourier accountant.

This algorithm is described in
Optimal Accounting of Differential Privacy via Characteristic Function.
https://arxiv.org/pdf/2106.08567.pdf

### Usage of Analytical Fourier Accountant

1. We describe each mechanism using a pair of log characteristic functions, corresponding to the dominating pair distribution.

    `from autodp.mechanism_zoo import GaussianMechanism`
    
    Declare a mechanism with a characteristic function-based description.
    
    `gm1 = GaussianMechanism(sigma, phi_off=False, name='phi_GM1')`

    
2. For composition, the ComposeAFA (from transformer_zoo) takes in a sequence of mechanisms to compose, and output the composed
mechanism. 

    `from autodp.transformer_zoo import  ComposeAFA`
    
    Define an analytical Fourier Accountant(AFA).
    
    `compose = ComposeAFA()`
    
    Compose the Gaussian mechanism for 10 times.
    
    `composed_mech_afa = compose([gm1], [10])`
    

3. To convert back to standard DP

    Computes the DP epsilon for a fixed delta   
    
    `eps_afa = composed_mech_afa.get_approxDP(delta)`
    
    Computes the DP delta for a fixed epsilon 
    
    `delta_afa = composed_mech_afa.get_approxDP(eps)`
    
    In more details, to convert back to delta(eps) or eps(delta), we will (1) use numerical inversion to
convert the characteristic function back to CDFs. (2) convert CDF to (epsilon, delta)-DP.

4. Amplification by sampling.

    The transformer takes in a base mechanism and returns the sampled mechanism for remove_only or add_only
 relationship).
 
    `from autodp.transformer_zoo import AmplificationBySampling_pld`
    
     Declare two transformers.
     
    `transformer_remove_only = AmplificationBySampling_pld(PoissonSampling=True, neighboring='remove_only')`
    
    `transformer_add_only = AmplificationBySampling_pld(PoissonSampling=True, neighboring='add_only')`
    
     Base mechanism is gaussian mechanism, sampling with probability prob.
     
    `sample_gau_remove_only =transformer_remove_only(gm1, prob)`
    
    `sample_gau_add_only =transformer_add_only(gm1, prob)`
    
     Obtain the standard DP by a pointwise maximum of two mechanisms.
     
    `sample_eps = max(sample_gau_remove_only.get_approxDP(delta), sample_gau_add_only.get_approxDP(delta) )`
 
    


### New features in API
``mechanism_zoo``: implements popular DP mechanisms with their privacy guarantees.

``converter``: implements all known conversions from DP, e.g., `phi_to_cdf`, `cdf_to_approxDP`.

``phi_bank``: A bank of phi-function based description.
 
``transformer_bank``: a callable object that takes one or more mechanism as input and **transform** them into a new mechanism.
For example: 

        ComposeAFA : supports characteristic function (phi-function) based composition.
        classAmplificationBySampling_pld: amplification by sampling for privacy loss distribution.


### Usage Example
`afa_simple_example.py`: Tutorials of AFA with basic mechanisms.

    1. Composition of Gaussian mechanism
    
    2. Composition of a heterogeneous sequence of mechanisms 

`afa_subsample.py`: Composition of Poisson Subsampled Gaussian mechanisms using analytical Fourier Accountant (AFA).
