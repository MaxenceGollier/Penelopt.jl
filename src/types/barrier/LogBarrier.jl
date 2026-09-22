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
  z_l::V
  z_u::V

  function LogBarrier(μ::T, l::V, u::V) where {T<:Real,V<:AbstractVector{T}}
    @assert μ > 0 "Barrier parameter μ must be positive."
    @assert length(l) == length(u) && all(l .< u) "Bounds must satisfy l < u."
    z_l, z_u = similar(l), similar(u)
    return new{T,V}(μ, l, u, z_l, z_u)
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

"`z_l[i], z_u[i] = μ / (x[i]-l[i]), μ / (u[i]-x[i])`"
function update_multipliers!(ϕ::LogBarrier, x)
  for i in eachindex(x)
    ϕ.z_l[i] = ϕ.μ * invd(ϕ.l[i], x[i] - ϕ.l[i])
    ϕ.z_u[i] = ϕ.μ * invd(ϕ.u[i], ϕ.u[i] - x[i])
  end
  return ϕ
end

# Needs to be called after `update_multipliers!`.
"`g += z_u - z_l`"
function add_grad!(g, ϕ::LogBarrier, x)
  for i in eachindex(g)
    g[i] += ϕ.z_u[i] - ϕ.z_l[i]
  end
  return g
end

"`h = α * diag(z_l/(x-l) + z_u/(u-x))`"
function hess_diag!(h, ϕ::LogBarrier, x, α = one(ϕ.μ))
  for i in eachindex(h)
    h[i] = α * (ϕ.z_l[i] * invd(ϕ.l[i], x[i] - ϕ.l[i]) + ϕ.z_u[i] * invd(ϕ.u[i], ϕ.u[i] - x[i]))
  end
  return h
end

isinterior(ϕ::LogBarrier, x) = all(i -> ϕ.l[i] < x[i] < ϕ.u[i], eachindex(x))
