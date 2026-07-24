problem_names = ["BT1", "MSS1", "SSINE", "VANDANIUMS"]
expected_status = [:first_order, :first_order, :infeasible, :infeasible]

tol = 1e-3

function test_problem(name, primal_solution, dual_solution, expected_status; linear_solver::String = "ldlt")
  nlp = CUTEstModel(name)

  # Test with R2
  @testset "NullHessian" begin
    null_model = NullHessianModel(nlp)
    stats = L2Penalty(null_model, atol = tol, rtol = tol)

    # Test whether the outputs are well defined
    @test stats.status == expected_status
    if expected_status == :first_order
      @test norm(primal_solution - stats.solution) ≤ 100*tol
      @test abs(stats.objective - obj(nlp, primal_solution)) ≤ 100*tol
      @test norm(stats.multipliers - dual_solution) ≤ 100*tol
      @test norm(
        jtprod(nlp, stats.solution, stats.multipliers) + grad(nlp, stats.solution),
        Inf,
      ) ≤ 100*tol
    end
    @test stats.primal_feas == norm(cons(nlp, stats.solution), Inf)

    # Test stability and allocations
    solver = L2PenaltySolver(null_model)
    stats_optimized = ExactPenaltyExecutionStats(null_model)
    @test @wrappedallocs(
      solve!(solver, null_model, stats_optimized, atol = 1e-3, rtol = 1e-3)
    ) == 0

    # Test that the second calling form gives the same output
    @test stats_optimized.status == stats.status
    @test stats_optimized.objective == stats.objective
    @test stats_optimized.primal_feas == stats.primal_feas
    @test stats_optimized.dual_feas == stats.dual_feas
    @test all(stats_optimized.multipliers .== stats.multipliers)
    @test all(stats_optimized.solution .== stats.solution)
    @test stats_optimized.iter == stats.iter
  end

  # Test with BFGS
  @testset "BFGS" begin
    LBFGS_model = CompactBFGSModel(nlp)
    stats = L2Penalty(LBFGS_model, atol = tol, rtol = tol; linear_solver = linear_solver)

    @test stats.status == expected_status
    if expected_status == :first_order
      @test norm(primal_solution - stats.solution) ≤ 100*tol
      @test abs(stats.objective - obj(nlp, primal_solution)) ≤ 100*tol
      @test norm(stats.multipliers - dual_solution) ≤ 100*tol
      @test norm(
        jtprod(nlp, stats.solution, stats.multipliers) + grad(nlp, stats.solution),
        Inf,
      ) ≤ 100*tol
    end
    @test abs(stats.primal_feas - norm(cons(nlp, stats.solution), Inf)) ≤ 1000*tol
    @test stats.solver_specific[:n_fact] > 0

    # Test stability and allocations
    NLPModels.reset!(LBFGS_model)
    solver = L2PenaltySolver(LBFGS_model, linear_solver = linear_solver)
    stats_optimized = ExactPenaltyExecutionStats(LBFGS_model)
    solve!(solver, LBFGS_model, stats_optimized, atol = 1e-3, rtol = 1e-3)

    # Test that the second calling form gives the same output
    @test stats_optimized.status == stats.status
    @test stats_optimized.objective == stats.objective
    @test stats_optimized.primal_feas == stats.primal_feas
    @test stats_optimized.dual_feas == stats.dual_feas
    @test all(stats_optimized.multipliers .== stats.multipliers)
    @test all(stats_optimized.solution .== stats.solution)
    @test stats_optimized.iter == stats.iter
    @test stats_optimized.solver_specific[:n_fact] == stats.solver_specific[:n_fact]
  end

  @testset "Exact" begin

    stats = L2Penalty(nlp, atol = tol, rtol = tol; linear_solver = linear_solver)

    # Test whether the outputs are well defined
    @test stats.status == expected_status
    if expected_status == :first_order
      @test norm(primal_solution - stats.solution) ≤ 100*tol
      @test abs(stats.objective - obj(nlp, primal_solution)) ≤ 100*tol
      @test norm(stats.multipliers - dual_solution) ≤ 100*tol
      @test norm(
        jtprod(nlp, stats.solution, stats.multipliers) + grad(nlp, stats.solution),
        Inf,
      ) ≤ 100*tol
    end
    @test abs(stats.primal_feas - norm(cons(nlp, stats.solution), Inf)) ≤ 1000*tol
    @test stats.solver_specific[:n_fact] > 0

    # Test stability and allocations
    solver = L2PenaltySolver(nlp, linear_solver = linear_solver)
    stats_optimized = ExactPenaltyExecutionStats(nlp)
    @test @wrappedallocs(solve!(solver, nlp, stats_optimized, atol = 1e-3, rtol = 1e-3)) ==
          0

    # Test that the second calling form gives the same output
    @test stats_optimized.status == stats.status
    @test stats_optimized.objective == stats.objective
    @test stats_optimized.primal_feas == stats.primal_feas
    @test stats_optimized.dual_feas == stats.dual_feas
    @test all(stats_optimized.multipliers .== stats.multipliers)
    @test all(stats_optimized.solution .== stats.solution)
    @test stats_optimized.iter == stats.iter
    @test stats_optimized.solver_specific[:n_fact] == stats.solver_specific[:n_fact]
  end

  finalize(nlp)
end
# Test a simple problem
@testset "BT1" begin
  primal_solution = [1, 0]
  dual_solution = [-99.5]
  linear_solver = !isnothing(Base.get_extension(ExactPenalty, :ExactPenaltyMUMPSExt)) ? "mumps" : "ldlt"
  test_problem("BT1", primal_solution, dual_solution, :first_order; linear_solver = linear_solver)
end

# Test a problem where the function f is unbounded from below
@testset "MARATOS" begin
  primal_solution = [1, 0]
  dual_solution = [0.499999]
  linear_solver = !isnothing(Base.get_extension(ExactPenalty, :ExactPenaltyMUMPSExt)) ? "mumps" : "ldlt"
  test_problem("MARATOS", primal_solution, dual_solution, :first_order; linear_solver = linear_solver)
end

# Test an infeasible problem
@testset "VANDANIUMS" begin
  if isnothing(Base.get_extension(ExactPenalty, :ExactPenaltyMUMPSExt))
    test_problem("VANDANIUMS", Float64[], Float64[], :infeasible)
  end
end

# Test a problem that requires the watchdog technique
@testset "SSINE" begin
  # The problem is infeasible but the primal feas is arbitrarily small for some ||x|| -> inf.
  # Only test that we converge to first order
  nlp = CUTEstModel("SSINE")
  stats = L2Penalty(nlp, atol = 1e-5, rtol = 0.0)
  @test stats.status == :first_order
  finalize(nlp)
end
# Test an ill-conditionned problem
# TODO: Add MSS1
