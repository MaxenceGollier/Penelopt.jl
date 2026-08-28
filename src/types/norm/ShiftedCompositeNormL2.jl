abstract type AbstractShiftedCompositeNorm end

@doc raw"""
    ShiftedCompositeNormL2(h, c!, J!, A, b)

Returns the shift of a function `c` composed with the `ℓ₂` norm (see CompositeNormL2.jl).
Here, `c` is linearized i.e, `c(x + s) ≈ c(x) + J(x)s`. 
```math
f(s) = λ ‖c(x) + J(x)s‖₂,
```
where `λ > 0`. `c!` and `J!` should implement functions 
```math
c : ℝⁿ ↦ ℝᵐ,
```
```math
J : ℝⁿ ↦ ℝᵐˣⁿ,   
```
such that `J` is the Jacobian of `c`. It is expected that `m ≤ n`.
`A` and `b` should respectively be a matrix and a vector which can respectively store the values of `J` and `c`.
`A` is expected to be sparse, `c!` and `J!` should have signatures
```
c!(b <: AbstractVector{Real}, xk <: AbstractVector{Real})
J!(A <: AbstractSparseMatrixCOO{Real, Integer}, xk <: AbstractVector{Real})
```
"""
mutable struct ShiftedCompositeNormL2{
  T<:Real,
  F0<:Function,
  F1<:Function,
  M<:AbstractMatrix{T},
  V<:AbstractVector{T},
} <: AbstractShiftedCompositeNorm
  h::NormL2{T}
  c!::F0
  J!::F1
  A::M
  b::V
  g::V
  function ShiftedCompositeNormL2(
    λ::T,
    c!::Function,
    J!::Function,
    A::AbstractMatrix{T},
    b::AbstractVector{T};
  ) where {T<:Real}
    if length(b) != size(A, 1)
      error(
        "ShiftedCompositeNormL2: Wrong input dimensions, there should be as many constraints as rows in the Jacobian",
      )
    end

    g = similar(b)

    new{T,typeof(c!),typeof(J!),typeof(A),typeof(b)}(
      NormL2(λ),
      c!,
      J!,
      A,
      b,
      g,
    )
  end
end

shifted(
  ψ::CompositeNormL2{T,F0,F1,M,V},
  xk::AbstractVector{T},
) where {T<:Real,F0<:Function,F1<:Function,M<:AbstractMatrix{T},V<:AbstractVector{T}} =
  begin
    A = ψ.A
    b = similar(ψ.b)
    ShiftedCompositeNormL2(
      ψ.h.lambda,
      ψ.c!,
      ψ.J!,
      A,
      b,
    )
  end

fun_name(ψ::ShiftedCompositeNormL2) = "shifted `ℓ₂` norm"
fun_expr(ψ::ShiftedCompositeNormL2) = "t ↦ ‖c(xk) + J(xk)t‖₂"
fun_params(ψ::ShiftedCompositeNormL2) = "c(xk) = $(ψ.b)\n" * " "^14 * "J(xk) = $(ψ.A)\n"

function (ψ::AbstractShiftedCompositeNorm)(y)
  mul!(ψ.g, ψ.A, y)
  ψ.g .+= ψ.b
  return ψ.h(ψ.g)
end

function shift!(
  ψ::AbstractShiftedCompositeNorm,
  shift::AbstractVector{R};
  J = nothing,
  c = nothing,
) where {R<:Real}
  isnothing(c) ? ψ.c!(ψ.b, shift) : (ψ.b .= c)
  isnothing(J) ? ψ.J!(ψ.A, shift) : (ψ.A.vals .= J.vals)
  return ψ
end
