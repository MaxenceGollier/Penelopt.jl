@doc """
Penelopt.jl: A Large-Scale Equality-Constrained Optimization Solver.

* 📖 Documentation: [https://MaxenceGollier.github.io/Penelopt.jl/stable](https://MaxenceGollier.github.io/Penelopt.jl/stable)
* 🗂️ Repository: [github.com/MaxenceGollier/Penelopt.jl](https://github.com/MaxenceGollier/Penelopt.jl)
* 💬 Discussions: [github.com/MaxenceGollier/Penelopt.jl/discussions](https://github.com/MaxenceGollier/Penelopt.jl/discussions)
* 🎯 Issues: [github.com/MaxenceGollier/Penelopt.jl/issues](https://github.com/MaxenceGollier/Penelopt.jl/issues)
"""
module Penelopt

using LinearAlgebra, Printf, SparseArrays
using NLPModels, NLPModelsModifiers
using LDLFactorizations, LinearOperators, QuadraticModels, SolverCore, SparseMatricesCOO

import SolverCore: get_status, reset!

abstract type AbstractPenalizedProblemSolver <: AbstractOptimizationSolver end

include("PeneloptExecutionStats.jl")

include("types/quasi-newton/NullHessian.jl")
include("types/quasi-newton/CompactBFGS.jl")

include("types/norm/NormL2.jl")
include("types/norm/CompositeNormL2.jl")
include("types/norm/ShiftedCompositeNormL2.jl")

include("types/pre-processing/FixedVariable.jl")
include("types/pre-processing/ShiftedConstraint.jl")

include("linear_algebra/K2.jl")
include("linear_algebra/construct_workspace.jl")
include("linear_algebra/ldlt.jl")

include("types/PenalizedProblem.jl")
include("types/ShiftedPenalizedProblem.jl")
include("types/Watchdog.jl")

include("subsolvers/more-sorensen.jl")

include("ir2n.jl")

include("algorithm.jl")

include("extrapolate.jl")
include("feas_computer.jl")
end
