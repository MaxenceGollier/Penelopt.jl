export PeneloptExecutionStats

function PeneloptExecutionStats(nlp::AbstractNLPModel{T,V}) where {T,V}
  stats = GenericExecutionStats(nlp, solver_specific = Dict{Symbol,Int}())
  set_solver_specific!(stats, :n_fact, T(0))
  return stats
end
