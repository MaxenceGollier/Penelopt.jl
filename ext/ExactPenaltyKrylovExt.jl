module ExactPenaltyKrylovExt

using Krylov, LinearOperators
using ExactPenalty

using LinearAlgebra

import ExactPenalty: AbstractKrylovWorkspace
import ExactPenalty: construct_minres_qlp_workspace, solve_system!, update_workspace!
import ExactPenalty: get_inertia, get_solution!, get_status
import ExactPenalty: set_dual_inertia!, set_primal_inertia!
import ExactPenalty: K2

include("Krylov/OpK2.jl")
include("Krylov/minres_qlp_workspace.jl")

end
