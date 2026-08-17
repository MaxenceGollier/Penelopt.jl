abstract type AbstractShiftedCompositeNorm end

@doc raw"""
    ShiftedCompositeNormL2(h, c!, J!, A, b; store_previous_jacobian::Bool = false)

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
Moreover, if you want shifted instances of the operator to store the previous Jacobian on each shift, you can specify `store_previous_jacobian = true`.
In this case, each time a shift is performed, the previous Jacobian is stored in the `A_prev` field.
This is particularly useful for quasi-Newton updates in the context of constrained optimization.
"""
mutable struct ShiftedCompositeNormL2{
  T<:Real,
  F0<:Function,
  F1<:Function,
  M<:AbstractMatrix{T},
  N<:Union{Nothing,M},
  V<:AbstractVector{T},
} <: AbstractShiftedCompositeNorm
  h::NormL2{T}
  c!::F0
  J!::F1
  A::M
  A_prev::N # (Optional) can be used to store the previous Jacobian, useful for quasi-Newton approximations
  b::V
  g::V
  function ShiftedCompositeNormL2(
    λ::T,
    c!::Function,
    J!::Function,
    A::AbstractMatrix{T},
    b::AbstractVector{T};
    store_previous_jacobian::Bool = false,
  ) where {T<:Real}
    if length(b) != size(A, 1)
      error(
        "ShiftedCompositeNormL2: Wrong input dimensions, there should be as many constraints as rows in the Jacobian",
      )
    end

    A_prev = store_previous_jacobian ? copy(A) : nothing
    g = similar(b)

    new{T,typeof(c!),typeof(J!),typeof(A),typeof(A_prev),typeof(b)}(
      NormL2(λ),
      c!,
      J!,
      A,
      A_prev,
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
      store_previous_jacobian = ψ.store_previous_jacobian,
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

function shift!(ψ::AbstractShiftedCompositeNorm, shift::AbstractVector{R}; J = nothing,c = nothing) where {R<:Real}
  !isnothing(ψ.A_prev) && (ψ.A_prev.vals .= ψ.A.vals) # Update previous Jacobian if necessary
  isnothing(c) ? ψ.c!(ψ.b, shift) : (ψ.b .= c)
  isnothing(J) ? ψ.J!(ψ.A, shift) : (ψ.A.vals .= J.vals)
  return ψ
end
