using JLD2

using CUTEst, NLPModelsIpopt, SolverBenchmark

problem_names = CUTEst.select_sif_problems(
  min_con = 1,
  only_equ_con = true,
  only_free_var = true,
  custom_filter = meta -> (
    meta["variables"]["number"] >= meta["constraints"]["number"] # Uncomment to allow problems with fixed variables
    #&& meta["variables"]["free"] + meta["variables"]["fixed"] == meta["variables"]["number"]
  )
)

# Speedup benchmark time for exact Hessian.
# Split problems across 2 runners.
split = parse(Int, get(ENV, "CUTEST_SPLIT", "1"))
n_splits = 2

@assert 1 <= split <= n_splits

problem_names = collect(problem_names)

n = length(problem_names)
first = fld((split - 1) * n, n_splits) + 1
last  = fld(split * n, n_splits)

problem_names = problem_names[first:last]

@info "Running CUTEst split $split/$n_splits: problems $first:$last ($(length(problem_names)) problems)"

problem_list = (CUTEstModel(name) for name in problem_names)

tol = 1e-6
max_time = 300.0

solvers = Dict(
  :ipopt_exact =>
    nlp -> ipopt(
      nlp,
      print_level = 0,
      tol = tol,
      dual_inf_tol = tol,
      constr_viol_tol = tol,
      compl_inf_tol = Inf,
      acceptable_iter = 0,
      s_max = 1e12,
      nlp_scaling_method = "none",
      max_cpu_time = max_time,
      max_iter = typemax(Int32),
    ),
)

stats = bmark_solvers(solvers, problem_list)
@save "benchmark/result/stats_ipopt_exact_$(split).jld2" stats