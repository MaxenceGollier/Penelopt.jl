@doc """
ExactPenalty.jl: A Large-Scale Equality-Constrained Optimization Solver.

* 📖 Documentation: [https://MaxenceGollier.github.io/ExactPenalty.jl/stable](https://MaxenceGollier.github.io/ExactPenalty.jl/stable)
* 🗂️ Repository: [github.com/MaxenceGollier/ExactPenalty.jl](https://github.com/MaxenceGollier/ExactPenalty.jl)
* 💬 Discussions: [github.com/MaxenceGollier/ExactPenalty.jl/discussions](https://github.com/MaxenceGollier/ExactPenalty.jl/discussions)
* 🎯 Issues: [github.com/MaxenceGollier/ExactPenalty.jl/issues](https://github.com/MaxenceGollier/ExactPenalty.jl/issues)
"""
module ExactPenalty

using LinearAlgebra, Printf, SparseArrays
using NLPModels, NLPModelsModifiers, RegularizedProblems
using LDLFactorizations,
  LinearOperators,
  ProximalOperators,
  QuadraticModels,
  ShiftedProximalOperators,
  SolverCore,
  SparseMatricesCOO

import SolverCore.reset!

abstract type AbstractPenalizedProblemSolver <: AbstractOptimizationSolver end

include("ExactPenaltyExecutionStats.jl")

include("types/quasi-newton/NullHessian.jl")
include("types/quasi-newton/CompactBFGS.jl")

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
