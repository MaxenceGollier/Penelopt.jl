using CUTEst, Penelopt, NLPModels, NLPModelsIpopt, NLPModelsModifiers, LinearAlgebra, DataFrames

import NLPModels: increment!

# Reproduces the exact kwargs of the corresponding benchmark script, so the
# candidate point x̄ is the actual point produced by the benchmarked run.
const BENCHMARK_TOL = 1e-6
const BENCHMARK_MAX_TIME = 300.0

const BENCHMARK_SOLVERS = Dict(
  (:l2penalty, :exact) =>
    nlp -> L2Penalty(
      nlp,
      print_level = 0,
      atol = BENCHMARK_TOL,
      rtol = 0.0,
      max_time = BENCHMARK_MAX_TIME,
      max_iter = typemax(Int),
      linear_solver = "mumps",
    ),
  (:l2penalty, :lbfgs) =>
    nlp -> L2Penalty(
      nlp,
      print_level = 0,
      atol = BENCHMARK_TOL,
      rtol = 0.0,
      max_time = BENCHMARK_MAX_TIME,
      max_iter = typemax(Int),
      qn_hessian_approximation = "bfgs",
      linear_solver = "mumps",
    ),
  (:ipopt, :exact) =>
    nlp -> ipopt(
      nlp,
      print_level = 0,
      tol = BENCHMARK_TOL,
      dual_inf_tol = BENCHMARK_TOL,
      constr_viol_tol = BENCHMARK_TOL,
      compl_inf_tol = Inf,
      acceptable_iter = 0,
      s_max = floatmax(Float64),
      nlp_scaling_method = "none",
      max_cpu_time = BENCHMARK_MAX_TIME,
      max_iter = typemax(Int32),
    ),
  (:ipopt, :lbfgs) =>
    nlp -> ipopt(
      nlp,
      print_level = 0,
      tol = BENCHMARK_TOL,
      dual_inf_tol = BENCHMARK_TOL,
      constr_viol_tol = BENCHMARK_TOL,
      compl_inf_tol = Inf,
      acceptable_iter = 0,
      s_max = floatmax(Float64),
      hessian_approximation = "limited-memory",
      nlp_scaling_method = "none",
      max_cpu_time = BENCHMARK_MAX_TIME,
      max_iter = typemax(Int32),
    ),
)

# ------------------------------------------------------------------------- #
# min_x 1/2 ||c(x)||² s.t. ||J(x̄)(x - x̄)|| ≤ Δ
# ------------------------------------------------------------------------- #
mutable struct TrustRegionNLS{S<:AbstractNLSModel} <: AbstractNLSModel{Float64,Vector{Float64}}
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

function NLPModels.jac_nln_coord!(M::TrustRegionNLS, x::AbstractVector, vals::AbstractVector)
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

"""
    check_local_infeasibility(nlp, xbar; Δ=10.0, tol=1e-9, feas_tol=1e-3)

Check whether `x̄ = xbar` is a locally infeasible point of `nlp` by solving

    min_x 1/2 ||c(x)||² s.t. ||J(x̄)(x - x̄)|| ≤ Δ

with IPOPT, starting from `x̄`. Certified infeasible if the trust-region
constraint is inactive at the solution and ||c(x)|| > feas_tol.

Returns `true`/`false` when conclusive, `missing` if the inner solve didn't
reach `:first_order` or stopped at the trust-region boundary.
"""
function check_local_infeasibility(
  nlp::AbstractNLPModel,
  xbar::AbstractVector;
  Δ = 10.0,
  tol = 1e-9,
  feas_tol = 1e-3,
)
  M = TrustRegionNLS(nlp, xbar, Δ)
  model = FeasibilityFormNLS(M)

  # r₀ = c(x̄), so the F(x) - r = 0 block is satisfied at x0
  x0 = vcat(xbar, cons(nlp, xbar))
  stats = ipopt(model, x0 = x0, tol = tol, print_level = 0)

  if stats.status != :first_order
    @warn "Local infeasibility check for $(nlp.meta.name) was inconclusive (inner IPOPT solve terminated with status $(stats.status))"
    return missing
  end

  n = nlp.meta.nvar
  xsol = stats.solution[1:n]
  tr_residual = norm(M.Jxbar * (xsol - xbar))
  primal_feas = norm(cons(nlp, xsol))

  if tr_residual >= Δ
    @warn "Local infeasibility check for $(nlp.meta.name) was inconclusive (trust-region constraint active at the solution; Δ = $Δ may be too small)"
    return missing
  end

  certified = primal_feas > feas_tol

  @debug "Local infeasibility check for $(nlp.meta.name): ||c(x̄)|| = $primal_feas, ||J(x̄)(x-x̄)|| = $tr_residual -> $(certified ? "certified infeasible" : "not certified")"

  return certified
end

"""
    certify_local_infeasibility(name, key)

Reproduce the run `key` (e.g. `:l2penalty_exact_current`, `:ipopt_exact`)
for CUTEst problem `name` and certify whether its point is locally
infeasible. Returns `true`/`false`/`missing` (see
[`check_local_infeasibility`](@ref)), or `missing` if the run can't be
reproduced.
"""
function certify_local_infeasibility(name::AbstractString, key::Symbol)
  parts = Symbol.(split(string(key), "_"))
  solver, hessian = parts[1], parts[2]

  solve_fn = get(BENCHMARK_SOLVERS, (solver, hessian), nothing)
  if solve_fn === nothing
    @warn "No known benchmark reproduction for key $(key); skipping $(name)."
    return missing
  end

  nlp = CUTEstModel(name)
  try
    x = try
      solve_fn(nlp).solution
    catch e
      @warn "Could not reproduce the $(key) run for $(name): $(e)"
      return missing
    end

    preprocess_nlp = nlp
    if length(nlp.meta.ifix) > 0
      preprocess_nlp = remove_fixed_variables(nlp)
      x = x[nlp.meta.ifree]
    end

    return check_local_infeasibility(preprocess_nlp, x)
  finally
    finalize(nlp)
  end
end

"""
    certify_own_infeasibility(stats, key)

Certify every `:infeasible` problem in `stats[key]` against its own point.
Used to precompute reference/ipopt certification once (see
certify-infeasibility.jl), instead of re-certifying fixed baselines on
every PR run.

Returns a DataFrame with columns `name`, `hessian`, `status`,
`certified_locally_infeasible`.
"""
function certify_own_infeasibility(stats::Dict{Symbol,DataFrame}, key::Symbol)
  df = stats[key]
  parts = Symbol.(split(string(key), "_"))
  hessian = parts[2]

  @info "Certifying infeasibility results for $(key)."

  rows = NamedTuple[]
  for i = 1:nrow(df)
    name = df[i, :name]
    status = df[i, :status]
    status != :infeasible && continue

    certified = certify_local_infeasibility(name, key)

    push!(
      rows,
      (name = name, hessian = hessian, status = status, certified_locally_infeasible = certified),
    )
  end

  return DataFrame(rows)
end
