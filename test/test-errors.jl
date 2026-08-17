# Test warnings, errors, and exceptions thrown by Penelopt.


@testset "Extensions not loaded" begin

  nlp = CUTEstModel("BT1")

  @test_warn "Penelopt.jl: MUMPS extension is not loaded. Please install MPI.jl and MUMPS.jl. Switching to LDLFactorizations.jl..." begin
    L2Penalty(nlp, linear_solver = "mumps")
  end
  @test_warn "Penelopt.jl: HSL extension is not functional. Please check your license and make sure you have loaded HSL_jll.jl appropriately. Switching to LDLFactorizations.jl..." begin
    L2Penalty(nlp, linear_solver = "ma57")
  end
  @test_warn "Penelopt.jl: Krylov extension is not loaded. Please install Krylov.jl. Switching to LDLFactorizations.jl..." begin
    L2Penalty(nlp, linear_solver = "minres_qlp")
  end

  finalize(nlp)
end

@testset "Wrong problem type" begin
end
