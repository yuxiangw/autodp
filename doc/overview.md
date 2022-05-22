# AutoDP v0.2

####Module

`Module mechanism_zoo`: Libralies of popular DP mechanisms inherited from Mechanism class.

`Module calibrator_zoo` Implements a number of ways to choose parameters of a mechanism to achieve a pre-defined privacy guarantee. All calibrators inherit from calibrator class.

`Module transformer_zoo` Implements a number of ways to convert one or multi-mechanisms to a new mechanism. All transformers inherit from transformer class.

####Classes

`class Mechanism`: An abstract base class  describes a randomized algorithm and its privacy properties.

`class Transformer`: A callable object that takes one or more mechanism as input and
    **transform** them into a new mechanism.
    
 `class Calibrator`: An abstract base class to calibrate noise (or other parameters) meet a pre-scribed privacy budget
 

####Mechanism zoos



|Mechanisms| Description | Input (parameters)|
| --- | ----------- |-----|
|GaussianMechanism|Supports RDP, f-DP, (eps,delta)-DP|`sigma` --- Gaussian parameter |
|ExactGaussianMechanism|The Gaussian mechanism with tight direct computation of everything| `sigma` --- Gaussian parameter|
|LaplaceMechanism |Supports RDP| `b` --- Laplace parameter|
|RandresponseMechanism|Supports RDP| `p` --- The Bernoulli probability|
|PureDP_Mechanism|Supports pureDP and RDP| `eps` --- the epsilon parameter|
|SubsampleGaussianMechanism|Supports RDP, the mechanism for calibrator with composed subsampled Gaussian mechanism|`prob` --- sample probability, `sigma`--- Gaussian parameter, `coeff` --- Number of composition|
|ComposedGaussianMechanism|The mechanism for calibrator with composed Gaussian mechanism |`sigma`--- Gaussian parameter, `coeff` --- Number of composition|
|NoisyScreenMechanism|Provides the data-dependent RDP of Noisy Screening (Theorem 7 in Private-kNN (Zhu et.al., CVPR-20)|`logp`, `logq` --- the log probability of passing the threshold over a pair of adjacent dataset |
|GaussianSVT_Mechanism|Provides RDP-based Gaussian sparse vector technique (Zhu et.al., NeurIPS-20)|`k`--- the maximum length before the SVT algorithm stops, `sigma` --- Gaussian parameter, 'c' --- the cut-off parameter for SVT |
|LaplaceSVT_Mechanism|Provides RDP-based Laplace sparse vector technique (Zhu et.al., NeurIPS-20)|`k`--- the maximum length before the SVT algorithm stops, `b` --- Laplace parameter, 'c' --- the cut-off parameter for SVT |
|StageWiseMechanism|Provides RDP-based stagewise generalized SVT (Zhu et.al., NeurIPS-20)|`k`--- the maximum length before the SVT algorithm stops, `sigma` --- Gaussian parameter, 'c' --- the cut-off parameter for SVT |
#### Transformer zoos
|Transformers|Description|Input|Return|
| --- | ----------- |-----|----|
|Composition|Support pureDP and RDP-based composition |Takes a list of mechanisms and number of times they appear|A new mechanism that represents the composed mechanism|
|ComposeGaussian|A specialized composition for only Gaussian mechanisms (optimal composition)|A list of Gaussian mechanism and number of times they appear|Composed Gaussian mechanism |
|AmplificationBySampling|Supports RDP-based poisson / replace_one sampling and their improved bounds|The basic mechanism before sampling is applied and the sampling probability|Subsampled mechanism|

#### Calibrator zoos
|Calibrator|Description|Input|Output|
| --- | ----------- |-----|----|
|eps_delta_calibrator|Calibrate the noise scale for one single mechanism based on RDP |mechanism to calibrate, the privacy budget and which noise parameter to optimize over|A new mechanism with the calibrated noise parameter|
|generalized_eps_delta_calibrator|Calibrate the noise scale for generalzied mechanisms (e.g., composed mechanisms) based on RDP|mechanism to calibrate, the privacy budget and which noise parameter to optimize over|A new mechanism with the calibrated noise parameter|
|ana_gaussian_calibrator| Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of Balle and Wang, ICML'18|Gaussian mechanism to calibrate, the privacy budget|A new mechanism with the calibrated noise parameter|