using JLD2

using MPI, MUMPS
using CUTEst, ExactPenalty, NLPModelsModifiers, SolverBenchmark

nmax = 10000
problem_names = CUTEst.select_sif_problems(
  min_con = 1,
  max_var = nmax,
  only_equ_con = true,
  only_free_var = true,
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
      linear_solver = "mumps"
    ),
)

stats = bmark_solvers(solvers, problem_list, skipif = nlp -> nlp.meta.ncon ≥ nlp.meta.nvar)
@save "benchmark/result/stats_exact.jld2" stats
