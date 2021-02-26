# Calibrator

A `calibrator`calibrates noise (or other parameters) meet a pre-scribed privacy budget
 
```
class Calibrator():
```



| Attributes| Description |
| --- | ----------- |
|eps_budget|   Pre-scribed epsilon budget|
|delta_budget|  Pre-scribed delta budget |
|obj_func|The objective function. The calibrator ouput a set of parameters that works while minimizing the obj_func as much as possible|
|calibrate|The lambda function to calibrate params for obj_func|

#### Member Functions 
| Function| Description |
| --- | ----------- |
|\__call__(self, *args, **kwargs) | Invokes the calibrate with the given arguments in the given context.|
