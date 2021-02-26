# Mechanism

A `mechanism` describes a randomized algorithm and its privacy properties.

```
class Mechanism():
```



| Attributes| Description |
| --- | ----------- |
|Approximate DP| epsilon as a function of delta |
|Renyi DP |  RDP epsilon as a function of \alpha |
|Approximate RDP|RDP conditioning on a failure probability delta0|
|f-DP|Type II error as a function of Type I error|
|epsilon| Pure DP bound|
|delta0|Failure probability which documents the delta to use for approximate RDP|
|local_flag|Indicates whether the guarantees are intended to be for local differential privacy|
|group_size|Integer measuring the granuality of DP.  Default is 1|
| replace_one|Flag indicating whether this is for add-remove definition or replace-one|

#### Member Functions 
| Function| Description |
| --- | ----------- |
|get_approxDP(self, delta)| Output epsilon as a function of delta  |
|get_approxRDP(self, delta)|Output epsilon as function of delta and alpha|
|get_RDP(self, apha)|Output RDP as a function of alpha|
|get_fDP(self, fpr)|Output false negative rate as a function of false positive rate|
|get_pureDP(self)|Output pure DP|
|get_eps(self, delta)|Outputs the smallest eps from multiple calculations|
|propogate_updates(self, func, type_of_update, delta0=0, BBGHS_conversion=True, fDP_based_conversion=False)|Receives a new description of the mechanisms and updates all functions. |
