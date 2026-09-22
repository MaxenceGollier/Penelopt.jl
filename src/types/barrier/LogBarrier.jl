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
  τ::T # Fraction to boundary
  τmin::T # Minimum fraction to boundary
  l::V
  u::V
  z_l::V
  z_u::V

  function LogBarrier(μ::T, l::V, u::V; τmin = T(0.99)) where {T<:Real,V<:AbstractVector{T}}
    @assert μ > 0 "Barrier parameter μ must be positive."
    @assert length(l) == length(u) && all(l .< u) "Bounds must satisfy l < u."
    z_l, z_u = similar(l), similar(u)
    τ = max(τmin, 1 - μ)
    return new{T,V}(μ, τ, τmin, l, u, z_l, z_u)
  end
end

@inline logterm(b, d) = isfinite(b) ? (d > 0 ? -log(d) : oftype(d, Inf)) : zero(d)
@inline invd(b, d) = isfinite(b) ? inv(d) : zero(d)

function set_fraction_to_boundary!(::Nothing) end
function set_fraction_to_boundary!(ϕ::LogBarrier)
  ϕ.τ = max(ϕ.τmin, 1 - ϕ.μ)
end

function set_barrier!(::Nothing) end
function set_barrier!(ϕ::LogBarrier, μ) where {T,S}
  ϕ.μ = μ
  set_fraction_to_boundary!(ϕ)
end

@doc raw"""
    truncate_to_boundary!(s, s_z, xk, zk_L, zk_U, ϕ::LogBarrier)

Apply the fraction-to-the-boundary rule (Wächter & Biegler, eq. (15)) so that the next
iterate stays strictly inside the bounds, and the bound multipliers stay strictly positive.

The largest primal step `α ∈ (0, 1]` is computed so that

    xk + α s - l ≥ (1 - τ)(xk - l)   and   u - xk - α s ≥ (1 - τ)(u - xk),

and, separately, the largest dual step `α_z ∈ (0, 1]` so that

    zk_L + α_z s_z_l ≥ (1 - τ) zk_L   and   zk_U + α_z s_z_u ≥ (1 - τ) zk_U,

where `τ = ϕ.τ`. Then `s` is scaled in place by `α` and `s_z = (s_z_l, s_z_u)` is scaled
in place by `α_z`. Infinite bounds are ignored. Returns `(α, α_z)`.
"""
function truncate_to_boundary!(
  s,
  s_z_l,
  s_z_u,
  xk,
  zk_L,
  zk_U,
  ϕ::LogBarrier{T},
) where {T}
  τ = ϕ.τ
  α = one(T)
  α_z = one(T)

  for i in eachindex(xk)
    l, u = ϕ.l[i], ϕ.u[i]

    if isfinite(l)
      # Primal: only a step towards l can hit the bound.
      if s[i] < 0
        α = min(α, -τ * (xk[i] - l) / s[i])
      end
      # Dual: only a decreasing multiplier can hit 0.
      if s_z_l[i] < 0
        α_z = min(α_z, -τ * zk_L[i] / s_z_l[i])
      end
    end

    if isfinite(u)
      # Primal: only a step towards u can hit the bound.
      if s[i] > 0
        α = min(α, τ * (u - xk[i]) / s[i])
      end
      if s_z_u[i] < 0
        α_z = min(α_z, -τ * zk_U[i] / s_z_u[i])
      end
    end
  end

  s .*= α
  s_z_l .*= α_z
  s_z_u .*= α_z

  return α, α_z
end

truncate_to_boundary!(s, s_z_l, s_z_u, xk, zk_L, zk_U, ::Nothing,) = (one(eltype(s)), one(eltype(s)))

function (ϕ::LogBarrier)(x)
  val = zero(eltype(x))
  for i in eachindex(x)
    val += logterm(ϕ.u[i], ϕ.u[i] - x[i]) + logterm(ϕ.l[i], x[i] - ϕ.l[i])
  end
  return ϕ.μ * val
end

function add_grad!(g, ϕ::LogBarrier, x)
  for i in eachindex(g)
    g[i] += ϕ.μ * invd(ϕ.l[i], x[i] - ϕ.l[i]) - ϕ.μ * invd(ϕ.u[i], ϕ.u[i] - x[i])
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

function get_z_l_step!(s_z_l, ϕ::Nothing, x, s_x) end
function get_z_u_step!(s_z_u, ϕ::Nothing, x, s_x) end

@doc raw"""
    get_z_l_step!(s_z_l, ϕ::LogBarrier, x, s_x)

Compute the primal-dual step for the lower-bound multipliers (Wächter & Biegler, eq. (12))
with slack ``x - l``:

    s_z_l = μ/(x-l) - z_l - z_l/(x-l) .* s_x

Entries for infinite lower bounds are set to zero.
"""
function get_z_l_step!(s_z_l, ϕ::LogBarrier, x, s_x)
  μ = ϕ.μ
  for i in eachindex(x)
    l = ϕ.l[i]
    if isfinite(l)
      d = inv(x[i] - l)
      s_z_l[i] = μ * d - ϕ.z_l[i] - ϕ.z_l[i] * d * s_x[i]
    else
      s_z_l[i] = zero(eltype(s_z_l))
    end
  end
  return s_z_l
end

@doc raw"""
    get_z_u_step!(s_z_u, ϕ::LogBarrier, x, s_x)

Compute the primal-dual step for the upper-bound multipliers (Wächter & Biegler, eq. (12))
with slack ``u - x``, whose step is ``-s_x``:

    s_z_u = μ/(u-x) - z_u + z_u/(u-x) .* s_x

Entries for infinite upper bounds are set to zero.
"""
function get_z_u_step!(s_z_u, ϕ::LogBarrier, x, s_x)
  μ = ϕ.μ
  for i in eachindex(x)
    u = ϕ.u[i]
    if isfinite(u)
      d = inv(u - x[i])
      s_z_u[i] = μ * d - ϕ.z_u[i] + ϕ.z_u[i] * d * s_x[i]
    else
      s_z_u[i] = zero(eltype(s_z_u))
    end
  end
  return s_z_u
end

get_z_l(::Nothing, T::Type) = T[]
get_z_u(::Nothing, T::Type) = T[]

get_z_l(ϕ::LogBarrier, T::Type) = ϕ.z_l
get_z_u(ϕ::LogBarrier, T::Type) = ϕ.z_u

isinterior(ϕ::LogBarrier, x) = all(i -> ϕ.l[i] < x[i] < ϕ.u[i], eachindex(x))
