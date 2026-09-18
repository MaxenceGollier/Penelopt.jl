abstract type AbstractBarrier end

@doc raw"""
    LogBarrier(μ, l, u)

Logarithmic barrier for the bounds `l ≤ x ≤ u`:

```math
ϕ(x) = -μ \sum_i [\log(u_i - x_i) + \log(x_i - l_i)],
```
where `μ > 0`. Infinite bounds are ignored, and `ϕ(x) = Inf` if `x` is not strictly feasible.
"""
mutable struct LogBarrier{T<:Real,V<:AbstractVector{T}} <: AbstractBarrier
  μ::T
  l::V
  u::V

  function LogBarrier(μ::T, l::V, u::V) where {T<:Real,V<:AbstractVector{T}}
    @assert μ > 0 "Barrier parameter μ must be positive."
    @assert length(l) == length(u) && all(l .< u) "Bounds must satisfy l < u."
    return new{T,V}(μ, l, u)
  end
end

@inline logterm(b, d) = isfinite(b) ? (d > 0 ? -log(d) : oftype(d, Inf)) : zero(d)
@inline invd(b, d) = isfinite(b) ? inv(d) : zero(d)

function (ϕ::LogBarrier)(x)
  val = zero(eltype(x))
  for i in eachindex(x)
    val += logterm(ϕ.u[i], ϕ.u[i] - x[i]) + logterm(ϕ.l[i], x[i] - ϕ.l[i])
  end
  return ϕ.μ * val
end

"`g += ∇ϕ(x)`"
function add_grad!(g, ϕ::LogBarrier, x)
  for i in eachindex(g)
    g[i] += ϕ.μ * (invd(ϕ.u[i], ϕ.u[i] - x[i]) - invd(ϕ.l[i], x[i] - ϕ.l[i]))
  end
  return g
end

"`h = α * diag(∇²ϕ(x))`"
function hess_diag!(h, ϕ::LogBarrier, x, α = one(ϕ.μ))
  for i in eachindex(h)
    h[i] = α * ϕ.μ * (invd(ϕ.u[i], ϕ.u[i] - x[i])^2 + invd(ϕ.l[i], x[i] - ϕ.l[i])^2)
  end
  return h
end
