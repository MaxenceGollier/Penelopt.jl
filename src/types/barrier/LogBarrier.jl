abstract type AbstractBarrier end

@doc raw"""
    LogBarrier(μ, l, u)

Returns function the logarithmic barrier function:
```math
f(x) = -μ * sum(log(u_i - x_i) + log(x_i - l_i)
```
where `μ > 0`.
"""
mutable struct LogBarrier{
  T <: Real
  V <: AbstractVector{T}
} <: AbstractBarrier
  μ::T
  l::V
  u::V
end

function LogBarrier(μ, l, u)
  @assert μ > 0 "Barrier parameter μ must be positive."
  @assert length(l) == length(u) "Lower and upper bounds must have the same length."

  @inbounds for i in eachindex(l, u)
    @assert l[i] != Inf "Lower bound l[$i] cannot be +Inf."
    @assert u[i] != -Inf "Upper bound u[$i] cannot be -Inf."
    @assert l[i] < u[i] "Invalid bounds at index $i: l[$i] >= u[$i]."
  end

  return LogBarrier(
    μ,
    l,
    u,
  )
end

function (ϕ::LogBarrier)(x)
  val = zero(eltype(x))

  for i in eachindex(x)
    if isfinite(ϕ.u[i])
      val -= ϕ.μ * log(ϕ.u[i] - x[i])
    end

    if isfinite(ϕ.l[i])
      val -= ϕ.μ * log(x[i] - ϕ.l[i])
    end
  end

  return val
end

