using MPI, MUMPS
using CUTEst, Penelopt, NLPModels, NLPModelsIpopt, NLPModelsModifiers, LinearAlgebra

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
#
# A trust-region-constrained feasibility model, centered at a fixed point
# x̄, with a *constant* Jacobian J(x̄) baked into the (single, quadratic)
# constraint. Only obj/grad/cons/jac are implemented: hess_available is set
# to false, so NLPModelsIpopt automatically falls back to IPOPT's
# limited-memory (L-BFGS) Hessian approximation for this diagnostic solve.
# ------------------------------------------------------------------------- #
mutable struct TrustRegionFeasibilityModel <: AbstractNLPModel{Float64,Vector{Float64}}
  meta::NLPModelMeta{Float64,Vector{Float64}}
  counters::Counters
  nlp::AbstractNLPModel
  xbar::Vector{Float64}
  Jxbar::Any
  Δ::Float64
end

function TrustRegionFeasibilityModel(nlp::AbstractNLPModel, xbar::AbstractVector, Δ::Real)
  n = nlp.meta.nvar
  Jxbar = jac(nlp, xbar)

  meta = NLPModelMeta(
    n;
    x0 = copy(xbar),
    lvar = nlp.meta.lvar,
    uvar = nlp.meta.uvar,
    ncon = 1,
    lcon = [-Inf],
    ucon = [Δ^2], # constraint is stored in its squared form, see cons!
    nnzj = n,
    hess_available = false,
    name = "TrustRegionFeasibility($(nlp.meta.name))",
  )

  return TrustRegionFeasibilityModel(meta, Counters(), nlp, Vector{Float64}(xbar), Jxbar, Float64(Δ))
end

function NLPModels.obj(model::TrustRegionFeasibilityModel, x::AbstractVector)
  increment!(model, :neval_obj)
  cx = cons(model.nlp, x)
  return 0.5 * dot(cx, cx)
end

function NLPModels.grad!(model::TrustRegionFeasibilityModel, x::AbstractVector, g::AbstractVector)
  increment!(model, :neval_grad)
  cx = cons(model.nlp, x)
  jtprod!(model.nlp, x, cx, g)
  return g
end

function NLPModels.cons!(model::TrustRegionFeasibilityModel, x::AbstractVector, c::AbstractVector)
  increment!(model, :neval_cons)
  # Stored as ||J(x̄)(x - x̄)||² ≤ Δ² rather than ||J(x̄)(x - x̄)|| ≤ Δ so
  # that the constraint (and its gradient) stays smooth at x = x̄, which is
  # exactly the starting point we solve from.
  Jd = model.Jxbar * (x - model.xbar)
  c[1] = dot(Jd, Jd)
  return c
end

function NLPModels.jac_structure!(
  model::TrustRegionFeasibilityModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  n = model.meta.nvar
  rows .= 1
  cols .= 1:n
  return rows, cols
end

function NLPModels.jac_coord!(
  model::TrustRegionFeasibilityModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  increment!(model, :neval_jac)
  Jd = model.Jxbar * (x - model.xbar)
  vals .= 2 .* (model.Jxbar' * Jd)
  return vals
end

"""
    check_local_infeasibility(nlp, xbar; Δ=10.0, tol=1e-9, feas_tol=1e-3)

Given a candidate point `x̄ = xbar` for `nlp`, check whether `x̄` is a
locally infeasible point by solving

    min_x 1/2 ||c(x)||² s.t. ||J(x̄)(x - x̄)|| ≤ Δ

with IPOPT (absolute tolerance `tol`), starting from `x̄`.

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
  model = TrustRegionFeasibilityModel(nlp, xbar, Δ)

  stats = ipopt(model, x0 = copy(xbar), tol = tol, print_level = 0)

  if stats.status != :first_order
    @warn "Local infeasibility check for $(nlp.meta.name) was inconclusive (inner IPOPT solve terminated with status $(stats.status))"
    return missing
  end

  xsol = stats.solution
  tr_residual = norm(model.Jxbar * (xsol - xbar))
  primal_feas = norm(cons(nlp, xsol))

  if tr_residual >= Δ
    @warn "Local infeasibility check for $(nlp.meta.name) was inconclusive (trust-region constraint active at the solution; Δ = $Δ may be too small)"
    return missing
  end

  certified = primal_feas > feas_tol

  @info "Local infeasibility check for $(nlp.meta.name): ||c(xsol)|| = $primal_feas, ||J(x̄)(x-x̄)|| = $tr_residual -> $(certified ? "certified infeasible" : "not certified")"

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
