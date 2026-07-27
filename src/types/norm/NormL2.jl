"""
    NormL2(λ=1)

With a nonnegative scalar parameter λ, return the ``L_2`` norm
```math
f(x) = λ\\cdot\\sqrt{x_1^2 + … + x_n^2}.
```
"""
mutable struct NormL2{T}
  lambda::T 
end

(f::NormL2)(x) = f.lambda * norm(x)