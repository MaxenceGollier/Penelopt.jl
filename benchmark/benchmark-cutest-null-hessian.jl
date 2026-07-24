using JLD2

using MPI, MUMPS
using CUTEst, Penelopt, NLPModelsModifiers, SolverBenchmark

nmax = 300
problem_names = CUTEst.select_sif_problems(
  min_con = 1,
  max_var = nmax,
  only_equ_con = true,
  only_free_var = true,
)
problem_list = (CUTEstModel(name) for name in problem_names)

tol = 1e-6

solvers = Dict(
  :l2penalty_r2 =>
    nlp -> L2Penalty(NullHessianModel(nlp), verbose = 0, atol = tol, rtol = 0.0),
)

stats = bmark_solvers(solvers, problem_list, skipif = nlp -> nlp.meta.ncon ≥ nlp.meta.nvar)
@save "benchmark/result/stats_r2.jld2" stats
