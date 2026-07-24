abstract type AbstractPenalizedProblem{T,S} <: AbstractRegularizedNLPModel{T,S} end

"""
    penalty_problem = L2PenalizedProblem(nlp::AbstractNLPModel)

Given an NLP model `nlp` representing the equality constrained problem

    minimize f(x) subject to c(x) = 0,

construct the L2 penalty problem

    minimize f(x) + τ‖c(x)‖₂.

The L2PenalizedProblem is made of the following components:
- `model`: the original NLP model `nlp`;
- `h`: the penalty term, which is a `CompositeNormL2` object, see `ShiftedProximalOperators.jl`.
"""
mutable struct L2PenalizedProblem{T,S,M<:AbstractNLPModel{T,S},H<:CompositeNormL2,meta<:AbstractNLPModelMeta} <:
               AbstractPenalizedProblem{T,S}
  model::M
  h::H
  meta::meta
end

# Constructors
function L2PenalizedProblem(nlp::AbstractNLPModel{T,S}) where {T,S}
  x0 = nlp.meta.x0

  # Allocating variables for the ShiftedProximalOperator structure
  (rows, cols) = jac_structure(nlp)
  vals = similar(rows, eltype(x0))
  A = SparseMatrixCOO(nlp.meta.ncon, nlp.meta.nvar, rows, cols, vals)
  b = similar(x0, eltype(x0), nlp.meta.ncon)

  store_previous_jacobian = isa(nlp, QuasiNewtonModel) ? true : false
  penalty = CompositeNormL2(
    one(T),
    (c, x) -> cons!(nlp, x, c),
    (j, x) -> jac_coord!(nlp, x, j.vals),
    A,
    b,
    store_previous_jacobian = store_previous_jacobian,
  )
  return L2PenalizedProblem(nlp, penalty, nlp.meta)
end

# NLPModels API
function NLPModels.neval_obj(nlp::L2PenalizedProblem)
  return NLPModels.neval_obj(nlp.model)
end

function NLPModels.get_ncon(nlp::L2PenalizedProblem)
  h = nlp.h
  return length(h.b)
end

# Miscellaneous
function set_penalty!(nlp::L2PenalizedProblem{T}, τ::T) where {T}
  nlp.h.h = NormL2(τ)
end
