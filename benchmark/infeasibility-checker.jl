using CUTEst, Penelopt, NLPModels, NLPModelsIpopt, NLPModelsModifiers, LinearAlgebra, DataFrames

import NLPModels: increment!

# Solver/Hessian-model combinations, reproducing *exactly* the kwargs used in
# the corresponding benchmark scripts (benchmark-cutest-exact-hessian.jl,
# benchmark-cutest-lbfgs.jl, benchmark-cutest-ipopt-exact-hessian.jl,
# benchmark-cutest-ipopt-lbfgs.jl), so that the candidate point x̄ we certify
# is the actual point produced by the benchmarked run, not a fresh solve.
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

# --- residual machinery: forward verbatim to FeasibilityResidual(nlp) ---
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
# FeasibilityFormNLS calls the *_nln (nonlinear-constraint) variants on the
# wrapped model directly - jac_structure!/jac_coord! are themselves built
# from jac_nln_structure!/jac_nln_coord! by NLPModels' generic API layer,
# not the other way around - so those are what must be implemented here
# (our one constraint is nonlinear, and since we never declare any `lin`
# indices in the meta, it's automatically classified as such). Hessian is
# not split this way (it's of the full Lagrangian), so hess_structure!/
# hess_coord! below stay unsuffixed.
function NLPModels.cons_nln!(M::TrustRegionNLS, x::AbstractVector, c::AbstractVector)
  increment!(M, :neval_cons_nln)
  # Stored as ||J(x̄)(x - x̄)||² ≤ Δ² rather than ||J(x̄)(x - x̄)|| ≤ Δ so
  # that the constraint (and its gradient) stays smooth at x = x̄, which is
  # exactly the starting point we solve from.
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
  # obj_weight is irrelevant here: FeasibilityFormNLS always calls this with
  # obj_weight = 0.0 (TrustRegionNLS's own "objective" - unused, since
  # FeasibilityFormNLS's objective is purely 1/2||r||² - never contributes),
  # so only the constraint's constant Hessian, weighted by y[1], matters.
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

Given a candidate point `x̄ = xbar` for `nlp`, check whether `x̄` is a
locally infeasible point by solving

    min_x 1/2 ||c(x)||² s.t. ||J(x̄)(x - x̄)|| ≤ Δ

with IPOPT (absolute tolerance `tol`), starting from `x̄`. Internally this
is built as `FeasibilityFormNLS(TrustRegionNLS(nlp, xbar, Δ))`, i.e. as the
(x, r) problem `min 1/2||r||² s.t. c(x) - r = 0, ||J(x̄)(x-x̄)||² ≤ Δ²`.

The problem is declared certifiably locally infeasible if, at the solution
`x` of the trust-region subproblem:
  - the trust-region constraint is *inactive*, i.e. ||J(x̄)(x - x̄)|| < Δ
    (so the result isn't just an artifact of the trust-region radius), and
  - ||c(x)|| > feas_tol (no nearby feasible point was found).

Returns:
  - `true`  if both conditions above hold (certified locally infeasible),
  - `false` if the trust-region constraint is inactive and ||c(x)|| ≤ feas_tol
    (a nearby point essentially satisfies the constraints),
  - `missing` if the result is inconclusive: either the inner IPOPT solve
    did not reach `:first_order`, or it stopped at the trust-region boundary
    (in which case Δ may be too small to tell).
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

  # (x, r) start: r₀ = c(x̄), so the F(x) - r = 0 block is satisfied at x0.
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

Reproduce the run identified by `key` (e.g. `:l2penalty_exact_current` or
`:ipopt_exact`) for the CUTEst problem `name`, using the exact same solver
and kwargs as the corresponding benchmark script, and certify whether the
resulting point x̄ is locally infeasible.

Returns `true`, `false`, or `missing` (see [`check_local_infeasibility`](@ref)).
Also returns `missing` if the run itself cannot be reproduced.
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

Certify every problem that the run identified by `key` (e.g.
`:l2penalty_exact` or `:ipopt_exact`) declared `:infeasible` in `stats[key]`,
against its own reproduced point - no pairing with another solver involved.

This is what precomputes the certification saved alongside the `reference`
(at `SaveBenchmark.yml` time) and `ipopt` (at `RunIpoptBenchmark.yml` time)
baselines, so that PR comparisons in `compare-benchmarks.jl` can *load*
those results instead of re-certifying two fixed baselines on every PR run.

Returns a DataFrame with columns `name`, `hessian`, `status`,
`certified_locally_infeasible` (`true`/`false`/`missing`, see
[`check_local_infeasibility`](@ref)).
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
