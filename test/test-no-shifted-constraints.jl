@testset "ShiftedConstraintModel" begin

  nvar = 5
  x0 = ones(nvar)

  # Construct a simple model with a shifted constraint:
  # min_x x₁² + ... + x₅²  s.t  x₁³ + ... + x₅³ = 5.
  f(x) = sum(x .^ 2)
  c(x) = sum(x .^ 3)
  lcon = [5.0]
  ucon = [5.0]
  nlp_shifted = ADNLPModel(f, x0, c, lcon, ucon)

  # Make sure the problem has a shifted constraint
  @test any(!iszero, nlp_shifted.meta.lcon) || any(!iszero, nlp_shifted.meta.ucon)

  # Construct a model with the constraint shift removed
  nlp_no_shift = remove_constraint_shift(nlp_shifted)

  # Make sure the constraint is no longer shifted
  @test all(iszero, nlp_no_shift.meta.lcon)
  @test all(iszero, nlp_no_shift.meta.ucon)

  # No-op on a problem with no shifted constraint
  @test remove_constraint_shift(nlp_no_shift) === nlp_no_shift

  c_no_shift_ad(x) = sum(x .^ 3) - 5
  lcon = [0.0]
  ucon = [0.0]
  nlp_no_shift_ad = ADNLPModel(f, x0, c_no_shift_ad, lcon, ucon)

  consistent_nlps([nlp_no_shift, nlp_no_shift_ad])
end
