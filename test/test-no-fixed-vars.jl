@testset "FixedVariableEliminationModel" begin

  nvar = 10
  x0 = ones(nvar)

  # Construct a simple model with fixed variables:
  # min_x x₁² + x₂² + x₃² + ...  s.t  x₁ = 1, x₃ = 1, x₅ = 1, ... and x₁³ + x₂³ + x₃³ + ... = 0.
  f(x) = sum(x .^ 2)
  lvar = [i % 2 == 1 ? 1.0 : -Inf for i = 1:10]
  uvar = [i % 2 == 1 ? 1.0 : Inf for i = 1:10]
  c(x) = sum(x .^ 3)
  lcon = [0.0]
  ucon = [0.0]
  nlp_fixed = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)

  # Make sure the problem has fixed variables
  @test length(nlp_fixed.meta.ifix) > 0

  # Construct a model with the fixed variables eliminated
  nlp_no_fixed = remove_fixed_variables(nlp_fixed)

  # Make sure the problem has no fixed variables
  @test length(nlp_no_fixed.meta.ifix) == 0

  # Construct the problem with fixed variables eliminated
  nvar_no_fixed = 5
  x0 = ones(nvar_no_fixed)

  f_no_fixed_ad(x) = sum(x .^ 2) + 5
  c_no_fixed_ad(x) = sum(x .^ 3) + 5
  lcon = [0.0]
  ucon = [0.0]
  nlp_no_fixed_ad = ADNLPModel(f_no_fixed_ad, x0, c_no_fixed_ad, lcon, ucon)

  consistent_nlps([nlp_no_fixed, nlp_no_fixed_ad])
end
