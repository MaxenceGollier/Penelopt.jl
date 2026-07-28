function more_sorensen_sigma!(
  solver::MoreSorensenSolver{T,V},
  reg_nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  stats::GenericExecutionStats{T,V,V};
  Δ::T = one(T),
) where {T,V,M,H,P}
  reset!(stats)
  
  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  solver_workspace = solver.workspace

  @views @. solver.u2[1:n]         = -solver.x1[1:n]
  @views @. solver.u2[(n+1):(n+m)] = 0
  solve_system!(solver_workspace, solver.u2) 
  get_solution!(solver.x2, solver_workspace)

  norm_x1 = norm(@view solver.x1[1:n])

  solver.u2 .= 0

  return @views norm_x1^2/dot(solver.x1[1:n], solver.x2[1:n])*(norm_x1/Δ - 1)
end