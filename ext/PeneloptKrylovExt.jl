module PeneloptKrylovExt

using Krylov, LinearOperators
using Penelopt

using LinearAlgebra

import Penelopt: AbstractKrylovWorkspace
import Penelopt: construct_minres_qlp_workspace, solve_system!, update_workspace!
import Penelopt: get_inertia, get_solution!, get_status
import Penelopt: set_dual_inertia!, set_primal_inertia!
import Penelopt: K2

include("Krylov/OpK2.jl")
include("Krylov/minres_qlp_workspace.jl")

end
