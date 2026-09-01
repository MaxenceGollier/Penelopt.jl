module PeneloptLDLFactorizationsExt

using LDLFactorizations
using Penelopt

using LinearAlgebra, SparseArrays, SparseMatricesCOO

import Penelopt: AbstractLDLTWorkspace
import Penelopt: construct_ldlt_workspace, solve_system!, update_workspace!
import Penelopt: get_inertia, get_solution!, get_status
import Penelopt: set_dual_inertia!, set_primal_inertia!

include("LDLFactorizations/ldlt.jl")

end
