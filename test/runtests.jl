using Penelopt
using ADNLPModels,
  CUTEst,
  Krylov,
  LinearOperators,
  NLPModels,
  NLPModelsModifiers,
  NLPModelsTest,
  QuadraticModels,
  SolverCore,
  SparseMatricesCOO

using LinearAlgebra, Random, SparseArrays, Test

import Penelopt: solve!, ShiftedCompositeNormL2

Random.seed!(0)

include("allocations-macro.jl")

include("instances/instance-reader.jl")
include("instances/instance-generator.jl")

@testset "pre-processing" begin
  include("test-no-fixed-vars.jl")
end

@testset "quasi-Newton" begin
  include("test-quasi-newton.jl")
end

@testset "Subsolvers" begin
  include("test-subsolvers.jl")
end

@testset "CUTEst-default" begin
  @test isnothing(Base.get_extension(Penelopt, :PeneloptMUMPSExt)) # Check that the extension is not loaded.
  include("test-cutest.jl")
end

@testset "Errors and warnings" begin
  include("test-errors.jl")
end

using MPI, MUMPS
@testset "CUTEst-MUMPS" begin
  @test !isnothing(Base.get_extension(Penelopt, :PeneloptMUMPSExt))
  include("test-cutest.jl")
end
