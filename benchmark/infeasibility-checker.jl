using MPI, MUMPS
using CUTEst, Penelopt, NLPModels, NLPModelsIpopt, NLPModelsModifiers, LinearAlgebra

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

"""
    check_local_infeasibility(nlp, x; kwargs...)

Given a candidate point `x̄ = x` for `nlp`, check whether `x` is a locally
infeasible point.

This converts

    min_x f(x) s.t. c(x) = 0

into

    min_{x,r} 1/2 ||r||² s.t. c(x) = r,

and solves it with IPOPT, starting from `x`, at high precision. If the
residual ||c(x)|| does not vanish at the solution of that problem, `x` is
declared a locally infeasible point.

Returns `true`/`false` when the check is conclusive (the inner IPOPT solve
reaches `:first_order`), and `missing` when it is not (e.g. the inner solve
hits `max_cpu_time` or otherwise fails to converge), meaning the result is
not certain.
"""
function check_local_infeasibility(
  nlp::AbstractNLPModel,
  x::AbstractVector;
  tol = 1e-12,
  max_cpu_time = 300.0,
  feas_tol = 1e-6,
)
  # Step 1: convert min_x f(x) s.t. c(x) = 0
  # into min_{x,r} 1/2 ||r||^2 s.t. c(x) = r, where r is a slack variable.

  # Step 1.1: convert min_x f(x) s.t. c(x) = 0 into min_{x,r} 1/2 ||c(x)||^2.
  feas_nls = FeasibilityResidual(nlp)

  # Step 1.2: convert min_{x,r} 1/2 ||c(x)||^2 into min_{x,r} 1/2 ||r||^2 s.t. c(x) = r.
  feas_nlp = FeasibilityFormNLS(feas_nls)

  # Step 2: check if the problem is locally infeasible at x.
  # For this, solve the problem min_{x,r} 1/2 ||r||^2 s.t. c(x) = r using IPOPT,
  # starting from x and with high precision.
  x0 = vcat(x, cons(nlp, x))
  stats = ipopt(feas_nlp, x0 = x0, tol = tol, max_cpu_time = max_cpu_time, print_level = 0)

  if stats.status != :first_order
    @warn "Local infeasibility check for $(nlp.meta.name) was inconclusive (inner IPOPT solve terminated with status $(stats.status))"
    return missing
  end

  # Step 3: the problem is certified locally infeasible if the residual
  # ||c(x)|| does not vanish at the solution.
  xsol = stats.solution[1:nlp.meta.nvar]
  primal_feas = norm(cons(nlp, xsol), Inf)
  certified = primal_feas > feas_tol

  @info "Local infeasibility check for $(nlp.meta.name): ||c(xsol)|| = $primal_feas -> $(certified ? "certified infeasible" : "not certified")"

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
