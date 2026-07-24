module ExactPenaltyHSLExt

using HSL
using ExactPenalty

using LinearAlgebra, SparseMatricesCOO

import ExactPenalty: AbstractHSLWorkspace
import ExactPenalty: construct_ma57_workspace, solve_system!, update_workspace!
import ExactPenalty: get_inertia, get_solution!, get_status
import ExactPenalty: set_dual_inertia!, set_primal_inertia!

function __init__()
  ExactPenalty._set_hsl_available(HSL.LIBHSL_isfunctional())
end

include("HSL/ma57_workspace.jl")

end
