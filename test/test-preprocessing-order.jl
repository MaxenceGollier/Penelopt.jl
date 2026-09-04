@testset "Preprocessing steps commute" begin

  nvar = 10
  x0 = ones(nvar)

  # A problem with BOTH fixed variables (x₁ = x₃ = ... = 1) AND a shifted
  # equality constraint (c(x) = 5 ≠ 0), so both preprocessing steps have
  # something to do and can be checked against each other regardless of
  # the order they are applied in.
  f(x) = sum(x .^ 2)
  lvar = [i % 2 == 1 ? 1.0 : -Inf for i = 1:nvar]
  uvar = [i % 2 == 1 ? 1.0 : Inf for i = 1:nvar]
  c(x) = sum(x .^ 3)
  lcon = [5.0]
  ucon = [5.0]
  nlp = ADNLPModel(f, x0, lvar, uvar, c, lcon, ucon)

  # Sanity check: this instance actually exercises both steps.
  @test length(nlp.meta.ifix) > 0
  @test any(!iszero, nlp.meta.lcon) || any(!iszero, nlp.meta.ucon)

  # Apply in both orders.
  nlp_fixed_then_shift = nlp |> remove_fixed_variables |> remove_constraint_shift
  nlp_shift_then_fixed = nlp |> remove_constraint_shift |> remove_fixed_variables

  # Both should have eliminated the fixed variables and reformulated the
  # shifted constraint down to c(x) = 0, independently of order.
  for m in (nlp_fixed_then_shift, nlp_shift_then_fixed)
    @test get_nvar(m) == nvar - length(nlp.meta.ifix)
    @test all(iszero, get_lcon(m))
    @test all(iszero, get_ucon(m))
  end

  # The two resulting models should behave identically.
  consistent_nlps([nlp_fixed_then_shift, nlp_shift_then_fixed])

  # Recovering the full solution must be correct regardless of which
  # wrapper ends up outermost.
  x_reduced = 2 * ones(get_nvar(nlp_fixed_then_shift))
  fixed_idx = nlp.meta.ifix
  fixed_val = nlp.meta.lvar[fixed_idx]

  for m in (nlp_fixed_then_shift, nlp_shift_then_fixed)
    x_full = recover_full_solution(m, x_reduced)
    @test length(x_full) == nvar
    @test x_full[fixed_idx] == fixed_val
  end

  @test recover_full_solution(nlp_fixed_then_shift, x_reduced) ==
        recover_full_solution(nlp_shift_then_fixed, x_reduced)
end
