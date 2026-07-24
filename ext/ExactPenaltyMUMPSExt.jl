module ExactPenaltyMUMPSExt

using MPI, MUMPS
using ExactPenalty

using LinearAlgebra, SparseMatricesCOO

import ExactPenalty: AbstractMUMPSWorkspace
import ExactPenalty: construct_mumps_workspace, solve_system!, update_workspace!
import ExactPenalty: get_inertia, get_solution!, get_status
import ExactPenalty: set_dual_inertia!, set_primal_inertia!

function __init__()
  MPI.Init()
end

include("MUMPS/mumps_workspace.jl")

end
