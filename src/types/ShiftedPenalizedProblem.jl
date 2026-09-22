export ShiftedL2PenalizedProblem

abstract type AbstractShiftedPenalizedProblem{T,S} <: AbstractPenalizedProblem{T,S} end

"""
    shifted_penalty_problem = ShiftedL2PenalizedProblem(nlp::L2PenalizedProblem, x::AbstractVector)

Given a `L2PenalizedProblem` model `nlp` representing the penalized problem

    minimize f(x) + τ‖c(x)‖₂,

construct the shifted L2 penalty problem around the point `x`:

    minimize f(x) + ∇f(x)ᵀ s + 1/2 sᵀ H(x) s + τ‖c(x) + J(x)s‖₂. 

The L2PenalizedProblem is made of the following components:
- `model`: the quadratic model of the original NLP model `nlp` around the point `x`;
- `h`: the shifted penalty term, which is a `ShiftedCompositeNormL2` object, see `jl`.
- `parent`: the original `L2PenalizedProblem` model `nlp`.
"""
mutable struct ShiftedL2PenalizedProblem{
  T,
  S,
  M<:AbstractQuadraticModel{T,S},
  H<:ShiftedCompositeNormL2,
  P<:Union{Nothing,L2PenalizedProblem},
  SN<:Union{Nothing,S},
  meta<:AbstractNLPModelMeta,
} <: AbstractShiftedPenalizedProblem{T,S}
  model::M
  h::H
  parent::P
  meta::meta
  _qn_y::SN
  _qn_x_prev::SN
  _is_first_shift::Bool
end

function ShiftedL2PenalizedProblem(
  penalty_nlp::L2PenalizedProblem{T,V},
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
) where {T,V,VN1<:Union{Nothing,V}, VN2<:Union{Nothing,V}}

  nlp, h = penalty_nlp.model, penalty_nlp.h
  ∇f = isnothing(∇f) ? grad(nlp, x) : ∇f
  ψ = shifted(h, x)

  # Quasi-Newton Constructor
  if !isnothing(find_model(QuasiNewtonModel, nlp))
    B = get_op(nlp)
    φ = QuadraticModel(∇f, B, x0 = x, regularize = true)

    return ShiftedL2PenalizedProblem(
      φ,
      ψ,
      penalty_nlp,
      penalty_nlp.meta,
      similar(∇f),
      zero(∇f),
      true,
    )
  else # Full Hessian Constructor
    n = length(x)
    y = isnothing(y) ? zeros(T, nlp.meta.ncon) : y

    Bi, Bj = hess_structure(nlp)
    Bv = hess_coord(nlp, x, y)
    B = SparseMatrixCOO(n, n, Bi, Bj, Bv)

    φ = QuadraticModel(∇f, B, x0 = x, regularize = true)

    ψ = shifted(h, x)

    return ShiftedL2PenalizedProblem(
      φ,
      ψ,
      penalty_nlp,
      penalty_nlp.meta,
      nothing,
      nothing,
      true,
    )
  end
end

# ShiftedProximalOperators API
function shifted(
  penalty_nlp::L2PenalizedProblem{T,V},
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
) where {T,V,VN1<:Union{Nothing,V}, VN2<:Union{Nothing,V}}
  return ShiftedL2PenalizedProblem(penalty_nlp, x; ∇f = ∇f, y = y)
end

# TODO: Move to QuadraticModels.jl ?
function shift!(
  φ::QuadraticModel{T,V,M},
  nlp::AbstractNLPModel{T,V},
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
) where {T,V,M <: SparseMatrixCOO,VN1<:Union{Nothing,V},VN2<:Union{Nothing,V}}
  g = φ.data.c
  isnothing(∇f) ? grad!(nlp, x, g) : (g .= ∇f)
  isnothing(y) ? hess_coord!(nlp, x, φ.data.H.vals) : hess_coord!(nlp, x, y, φ.data.H.vals)
end

function shift!(
  φ::QuadraticModel{T,V,M},
  nlp::AbstractNLPModel{T,V},
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
) where {T,V,M,VN1<:Union{Nothing,V},VN2<:Union{Nothing,V}}
  g = φ.data.c
  isnothing(∇f) ? grad!(nlp, x, g) : (g .= ∇f)
end

# A callback that specifies specific actions to be taken before a shift is performed on the shifted penalty problem.
function _pre_shift_cb!(
  shifted_penalty_nlp::ShiftedL2PenalizedProblem{T,V},
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
  J::Ma = nothing,
  c::VN3 = nothing,
) where {
  T,
  V,
  VN1<:Union{Nothing,V},
  VN2<:Union{Nothing,V},
  VN3<:Union{Nothing,V},
  Ma<:Union{Nothing,AbstractMatrix{T}},
}
  nlp, h = shifted_penalty_nlp.parent.model, shifted_penalty_nlp.parent.h
  φ, ψ = shifted_penalty_nlp.model, shifted_penalty_nlp.h
  qn_y, qn_x_prev = shifted_penalty_nlp._qn_y, shifted_penalty_nlp._qn_x_prev
  is_first_shift = shifted_penalty_nlp._is_first_shift

  # BarrierModel pre shift callback.
  barrier_nlp = find_model(LogBarrierModel, shifted_penalty_nlp.parent)
  if !isnothing(barrier_nlp)
    update_multipliers!(barrier_nlp.ϕ, x)
  end

  # CompactBFGSModel pre shift callback.
  if !isnothing(find_model(CompactBFGSModel, shifted_penalty_nlp.parent)) 
    qn_s = qn_x_prev
    g, B = φ.data.c, φ.data.H
    if !is_first_shift
      qn_y .= g
      if !isnothing(y)
        mul!(qn_y, ψ.A', y, -one(T), -one(T)) # y = - g_prev - J(x)_prev^T λ
      end
    end
  end

end

# A callback that specifies specific actions to be taken after a shift is performed on the shifted penalty problem.
function _post_shift_cb!(
  shifted_penalty_nlp::ShiftedL2PenalizedProblem{T,V},
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
  J::Ma = nothing,
  c::VN3 = nothing,
) where {
  T,
  V,
  VN1<:Union{Nothing,V},
  VN2<:Union{Nothing,V},
  VN3<:Union{Nothing,V},
  Ma<:Union{Nothing,AbstractMatrix{T}},
}
  nlp, h = shifted_penalty_nlp.parent.model, shifted_penalty_nlp.parent.h
  φ, ψ = shifted_penalty_nlp.model, shifted_penalty_nlp.h
  qn_y, qn_x_prev = shifted_penalty_nlp._qn_y, shifted_penalty_nlp._qn_x_prev
  is_first_shift = shifted_penalty_nlp._is_first_shift

  # CompactBFGSModel post shift callback.
  if !isnothing(find_model(CompactBFGSModel, shifted_penalty_nlp.parent)) 
    if !is_first_shift
      @. qn_y .+= g
      if !isnothing(y)
        mul!(qn_y, ψ.A', y, one(T), one(T)) # y = y + J(x)^T λ 
      end
      qn_s .= x .- qn_x_prev
      push!(B, qn_s, qn_y)
    else
      shifted_penalty_nlp._is_first_shift = false
    end
    qn_x_prev .= x
  end

end

function shift!(
  shifted_penalty_nlp::ShiftedL2PenalizedProblem{T,V},
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
  J::Ma = nothing,
  c::VN3 = nothing,
) where {
  T,
  V,
  VN1<:Union{Nothing,V},
  VN2<:Union{Nothing,V},
  VN3<:Union{Nothing,V},
  Ma<:Union{Nothing,AbstractMatrix{T}},
}
  nlp, h = shifted_penalty_nlp.parent.model, shifted_penalty_nlp.parent.h
  φ, ψ = shifted_penalty_nlp.model, shifted_penalty_nlp.h

  _pre_shift_cb!(
    shifted_penalty_nlp,
    x;
    ∇f = ∇f,
    y = y,
    J = J,
    c = c,
  )

  shift!(φ, nlp, x, ∇f = ∇f, y = y)
  shift!(ψ, x, J = J, c = c)

  _post_shift_cb!(
    shifted_penalty_nlp,
    x;
    ∇f = ∇f,
    y = y,
    J = J,
    c = c,
  )
end

# Miscellaneous
function set_penalty!(nlp::ShiftedL2PenalizedProblem{T}, τ::T) where {T}
  nlp.h.h.lambda = τ
  nlp.parent.h.h.lambda = τ
end

function check_descent(
  shifted_penalty_nlp::ShiftedL2PenalizedProblem{T},
  s::AbstractVector,
) where {T}
  φ, ψ = shifted_penalty_nlp.model, shifted_penalty_nlp.h

  cx, τ = ψ.b, ψ.h.lambda
  ψ0 = τ * norm(cx) # φ0 = 0
  return ψ0 - obj(φ, s) - ψ(s) >= 0
end

function reset!(
  shifted_penalty_nlp::ShiftedL2PenalizedProblem,
)
  nlp, h = shifted_penalty_nlp.parent.model, shifted_penalty_nlp.parent.h
  φ, ψ = shifted_penalty_nlp.model, shifted_penalty_nlp.h

  if !isnothing(find_model(CompactBFGSModel, shifted_penalty_nlp.parent)) 
     x_prev = shifted_penalty_nlp._qn_x_prev .= 0

    LinearOperators.reset!(φ.data.H)
    shifted_penalty_nlp._is_first_shift = true
  end
end