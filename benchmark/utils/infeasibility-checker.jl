using CUTEst, Penelopt, NLPModels, NLPModelsIpopt, NLPModelsModifiers, LinearAlgebra, DataFrames

include(joinpath(@__DIR__, "trust-region-nls.jl"))

# Needs benchmark-utils.jl included first (BENCHMARK_SOLVERS).

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

  if tr_residual >= Δ - tol
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
