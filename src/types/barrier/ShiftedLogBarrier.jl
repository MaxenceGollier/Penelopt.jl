abstract type AbstractShiftedBarrier end

@doc raw"""
    ShiftedLogBarrier(μ, l, u)

Second-order Taylor approximation of the logarithmic barrier around `x`:

```math
ϕ(x + s) ≈ ϕ(x) + ∇ϕ(x)'s + 1/2 s'∇²ϕ(x)s

where

ϕ(x) = -μ * ∑ᵢ [log(uᵢ - xᵢ) + log(xᵢ - lᵢ)].

Infinite bounds are treated as inactive constraints:
lᵢ = -Inf means no lower-barrier term and
uᵢ = Inf means no upper-barrier term.
"""
mutable struct ShiftedLogBarrier{
T <: Real,
V <: AbstractVector{T}
} <: AbstractShiftedBarrier
μ::T
l::V
u::V
end

function (ϕ::ShiftedLogBarrier)(x, s)
value = zero(eltype(x))

@inbounds for i in eachindex(x, ϕ.l, ϕ.u)
    xi = x[i]
    si = s[i]

    # Constant term: ϕ(x)
    if isfinite(ϕ.u[i])
        value -= ϕ.μ * log(ϕ.u[i] - xi)
    end

    if isfinite(ϕ.l[i])
        value -= ϕ.μ * log(xi - ϕ.l[i])
    end

    # First-order term: ∇ϕ(x)'s
    if isfinite(ϕ.u[i])
        value += ϕ.μ / (ϕ.u[i] - xi) * si
    end

    if isfinite(ϕ.l[i])
        value -= ϕ.μ / (xi - ϕ.l[i]) * si
    end

    # Second-order term: 1/2 s'∇²ϕ(x)s
    if isfinite(ϕ.u[i])
        value += ϕ.μ / (2 * (ϕ.u[i] - xi)^2) * si^2
    end

    if isfinite(ϕ.l[i])
        value += ϕ.μ / (2 * (xi - ϕ.l[i])^2) * si^2
    end
end

return value

end