module PeneloptMUMPSExt

using MPI, MUMPS
using Penelopt

using LinearAlgebra, SparseMatricesCOO

import Penelopt: AbstractMUMPSWorkspace
import Penelopt: construct_mumps_workspace, solve_system!, update_workspace!
import Penelopt: get_inertia, get_solution!, get_status
import Penelopt: set_dual_inertia!, set_primal_inertia!

function __init__()
  MPI.Init()
end

include("MUMPS/mumps_workspace.jl")

end
