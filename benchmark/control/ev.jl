using OptimalControlProblems, OptimalControl, Plots

using Penelopt, MUMPS, MPI

tol = 1e-6
max_time = 300.0

docp = chain(OptimalControlBackend())
nlp = nlp_model(docp)

stats = ipopt(
  nlp,
  print_level = 0,
  tol = tol,
)

reset!(nlp)

stats = ipopt(
  nlp,
  print_level = 0,
  tol = tol,
)

println(nlp.counters)
println(stats.elapsed_time)

NLPModels.reset!(nlp)

stats = L2Penalty(
  nlp,
  print_level = 0,
  atol = tol,
  rtol = 0.0,
  max_time = max_time,
  max_iter = typemax(Int),
  linear_solver = "ldlt",
)

println(nlp.counters)
println(stats.elapsed_time)

ocp_sol = build_OCP_solution(docp, stats)

# dimensions
n = state_dimension(ocp_sol)
m = control_dimension(ocp_sol)

plot(
  ocp_sol;
  color=1,
  size=(816, 240*(n+m)),
  label="OptimalControl",
  #leftmargin=20mm,
)

