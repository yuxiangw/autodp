# Transformer

A `transformer`is a callable object that takes one or more mechanism as input and
    **transform** them into a new mechanism

```
class Transformer():
```



| Attributes| Description |
| --- | ----------- |
|unary_operator|  If true it takes one mechanism as an input, otherwise takes many mechanisms as an input|
|preprocessing|  It specifies whether the operation is fore or after the mechanism,  e.g., amplification by sampling is before applying the mechanism |

#### Member Functions 
| Function| Description |
| --- | ----------- |
|\__call__(self, *args, **kwargs) | Invokes the computation with the given arguments in the given context. |

#### Examples of Transformer

| Instantiation | Description | Feature |
| --- | -------------------------- | --- |
| Composition | The generic composition class that supports RDP-based composition | RDP-based composition |

