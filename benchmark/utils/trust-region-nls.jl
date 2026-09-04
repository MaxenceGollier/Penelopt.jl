using NLPModels, NLPModelsModifiers, LinearAlgebra

import NLPModels: increment!

# ------------------------------------------------------------------------- #
# min_x 1/2 ||c(x)||² s.t. ||J(x̄)(x - x̄)|| ≤ Δ
# ------------------------------------------------------------------------- #
mutable struct TrustRegionNLS{S<:AbstractNLSModel} <:
               AbstractNLSModel{Float64,Vector{Float64}}
  meta::NLPModelMeta{Float64,Vector{Float64}}
  nls_meta::NLSMeta{Float64,Vector{Float64}}
  counters::NLSCounters
  feas_nls::S
  xbar::Vector{Float64}
  Jxbar::Any
  Hxbar::Matrix{Float64} # constant Hessian of the constraint: 2 J(x̄)ᵀJ(x̄)
  Δ::Float64
end

function TrustRegionNLS(nlp::AbstractNLPModel, xbar::AbstractVector, Δ::Real)
  feas_nls = FeasibilityResidual(nlp)
  n = nlp.meta.nvar
  Jxbar = jac(nlp, xbar)
  Hxbar = 2 .* Matrix(Jxbar' * Jxbar)

  meta = NLPModelMeta(
    n;
    x0 = copy(xbar),
    lvar = feas_nls.meta.lvar,
    uvar = feas_nls.meta.uvar,
    ncon = 1,
    lcon = [-Inf],
    ucon = [Δ^2], # constraint is stored in its squared form, see cons!
    nnzj = n,
    nnzh = div(n * (n + 1), 2),
    name = "TrustRegionNLS($(nlp.meta.name))",
  )

  return TrustRegionNLS(
    meta,
    feas_nls.nls_meta,
    NLSCounters(),
    feas_nls,
    Vector{Float64}(xbar),
    Jxbar,
    Hxbar,
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

# --- our one extra general constraint: ||J(x̄)(x - x̄)||² ≤ Δ² ---
#
# FeasibilityFormNLS calls the *_nln variants directly, not cons!/jac_*!.
# Hessian isn't split this way, so hess_structure!/hess_coord! stay unsuffixed.
function NLPModels.cons_nln!(M::TrustRegionNLS, x::AbstractVector, c::AbstractVector)
  increment!(M, :neval_cons_nln)
  # squared form keeps the constraint gradient smooth at x = x̄ (start point)
  Jd = M.Jxbar * (x - M.xbar)
  c[1] = dot(Jd, Jd)
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
  vals .= 2 .* (M.Jxbar' * Jd)
  return vals
end

function NLPModels.hess_structure!(
  M::TrustRegionNLS,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  n = M.meta.nvar
  idx = 1
  for j = 1:n, i = j:n
    rows[idx] = i
    cols[idx] = j
    idx += 1
  end
  return rows, cols
end

function NLPModels.hess_coord!(
  M::TrustRegionNLS,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight = 1.0,
)
  # obj_weight always 0 here (FeasibilityFormNLS's own objective is 1/2||r||²)
  increment!(M, :neval_hess)
  n = M.meta.nvar
  yc = length(y) > 0 ? y[1] : 0.0
  idx = 1
  for j = 1:n, i = j:n
    vals[idx] = yc * M.Hxbar[i, j]
    idx += 1
  end
  return vals
end
