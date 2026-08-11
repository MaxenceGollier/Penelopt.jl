using JLD2

using MPI, MUMPS
using CUTEst, Penelopt, NLPModelsModifiers, SolverBenchmark

problem_names = CUTEst.select_sif_problems(
  min_con = 1,
  only_equ_con = true,
  only_free_var = true,
  custom_filter = meta -> (
    meta["variables"]["number"] >= meta["constraints"]["number"] # Uncomment to allow problems with fixed variables
    #&& meta["variables"]["free"] + meta["variables"]["fixed"] == meta["variables"]["number"]
  )
)
problem_list = (CUTEstModel(name) for name in problem_names)

tol = 1e-6
max_time = 300.0

solvers = Dict(
  :l2penalty_exact =>
    nlp -> L2Penalty(
      nlp,
      print_level = 0,
      atol = tol,
      rtol = 0.0,
      max_time = max_time,
      max_iter = typemax(Int),
      linear_solver = "mumps",
    ),
)

stats = bmark_solvers(solvers, problem_list)
@save "benchmark/result/stats_exact.jld2" stats
