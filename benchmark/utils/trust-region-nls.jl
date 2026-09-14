using NLPModels, NLPModelsModifiers, LinearAlgebra, SparseArrays

import NLPModels: increment!

# ------------------------------------------------------------------------- #
# min_x 1/2 ||c(x)||² s.t. ||J(x̄)(x - x̄)||² ≤ Δ²
# ------------------------------------------------------------------------- #
mutable struct TrustRegionNLS{T,V,S<:AbstractNLSModel,M} <: AbstractNLSModel{T,V}
  meta::NLPModelMeta{T,V}
  nls_meta::NLSMeta{T,V}
  counters::NLSCounters
  feas_nls::S
  xbar::V
  Jxbar::M
  Hxbar_rows::Vector{Int}
  Hxbar_cols::Vector{Int}
  Hxbar_vals::V
  Δ::T
end

function TrustRegionNLS(nlp::AbstractNLPModel, xbar::AbstractVector, Δ::Real)
  feas_nls = FeasibilityResidual(nlp)
  n = nlp.meta.nvar
  Jxbar = jac(nlp, xbar)
  Hxbar = LinearAlgebra.tril(2 .* (Jxbar' * Jxbar))
  Hxbar_rows, Hxbar_cols, Hxbar_vals = findnz(Hxbar)

  meta = NLPModelMeta(
    n;
    x0 = copy(xbar),
    lvar = feas_nls.meta.lvar,
    uvar = feas_nls.meta.uvar,
    ncon = 1,
    lcon = [-Inf],
    ucon = [Δ^2/2], # constraint is stored in its squared form, see cons!
    nnzj = n,
    nnzh = length(Hxbar_vals),
    name = "TrustRegionNLS($(nlp.meta.name))",
  )

  return TrustRegionNLS(
    meta,
    feas_nls.nls_meta,
    NLSCounters(),
    feas_nls,
    Vector{Float64}(xbar),
    Jxbar,
    Hxbar_rows,
    Hxbar_cols,
    Hxbar_vals,
    Float64(Δ),
  )
end

# --- forward residual machinery to FeasibilityResidual(nlp) ---
NLPModels.residual!(M::TrustRegionNLS, x::AbstractVector, Fx::AbstractVector) =
  residual!(M.feas_nls, x, Fx)
NLPModels.jac_structure_residual!(
  M::TrustRegionNLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) = jac_structure_residual!(M.feas_nls, rows, cols)
NLPModels.jac_coord_residual!(M::TrustRegionNLS, x::AbstractVector, vals::AbstractVector) =
  jac_coord_residual!(M.feas_nls, x, vals)
NLPModels.jprod_residual!(
  M::TrustRegionNLS,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
) = jprod_residual!(M.feas_nls, x, v, Jv)
NLPModels.jtprod_residual!(
  M::TrustRegionNLS,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
) = jtprod_residual!(M.feas_nls, x, v, Jtv)
NLPModels.hess_structure_residual!(
  M::TrustRegionNLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) = hess_structure_residual!(M.feas_nls, rows, cols)
NLPModels.hess_coord_residual!(
  M::TrustRegionNLS,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
) = hess_coord_residual!(M.feas_nls, x, v, vals)

# --- one extra general constraint: ||J(x̄)(x - x̄)||²/2 ≤ Δ²/2 ---
#
# FeasibilityFormNLS calls the *_nln variants directly, not cons!/jac_*!.
# Hessian isn't split this way, so hess_structure!/hess_coord! stay unsuffixed.
function NLPModels.cons_nln!(M::TrustRegionNLS, x::AbstractVector, c::AbstractVector)
  increment!(M, :neval_cons_nln)
  # squared form keeps the constraint gradient smooth at x = x̄ (start point)
  Jd = M.Jxbar * (x - M.xbar)
  c[1] = dot(Jd, Jd) / 2
  return c
end

function NLPModels.jac_nln_structure!(
  M::TrustRegionNLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  n = M.meta.nvar
  rows .= 1
  cols .= 1:n
  return rows, cols
end

function NLPModels.jac_nln_coord!(
  M::TrustRegionNLS,
  x::AbstractVector,
  vals::AbstractVector,
)
  increment!(M, :neval_jac_nln)
  Jd = M.Jxbar * (x - M.xbar)
  vals .= (M.Jxbar' * Jd)
  return vals
end

function NLPModels.hess_structure!(
  M::TrustRegionNLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= M.Hxbar_rows
  cols .= M.Hxbar_cols
  return rows, cols
end

function NLPModels.hess_coord!(
  M::TrustRegionNLS,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  increment!(M, :neval_hess)
  yc = length(y) > 0 ? y[1] : zero(T)
  vals .= yc .* M.Hxbar_vals
  return vals
end
