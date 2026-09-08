@testset "ScaledModel" begin

  nvar = 5
  x0 = ones(nvar)
  d_f = 2.0
  d_c = [3.0, 0.5]

  f(x) = sum(x .^ 2)
  c(x) = [sum(x .^ 3), sum(x)]
  lcon = [1.0, 2.0]
  ucon = [1.0, 2.0]

  # Hand-written equivalent of the scaled problem
  f_scaled(x) = d_f * f(x)
  c_scaled(x) = d_c .* c(x)

  # consistent_nlps drives calls on both models and checks their counters
  # end up equal, so each pair below needs its own fresh, otherwise-unused
  # models: a ScaledModel's calls also increment the wrapped model's own
  # counters, so reusing a model across two comparisons throws counters off.
  nlp_scaled = scale_model(ADNLPModel(f, x0, c, lcon, ucon); d_f = d_f, d_c = d_c)
  nlp_scaled_ad = ADNLPModel(f_scaled, x0, c_scaled, d_c .* lcon, d_c .* ucon)
  consistent_nlps([nlp_scaled, nlp_scaled_ad])

  # Default scaling factors leave the problem unchanged. Compare against
  # scale_model wrapping a *separate* instance: every call consistent_nlps
  # makes to the wrapper also bumps the wrapped model's own counters, so
  # wrapping the same instance being compared against would double-count it.
  consistent_nlps([ADNLPModel(f, x0, c, lcon, ucon), scale_model(ADNLPModel(f, x0, c, lcon, ucon))])

  # unscale/scale round-trips
  nlp = ADNLPModel(f, x0, c, lcon, ucon)
  nlp_scaled = scale_model(nlp; d_f = d_f, d_c = d_c)
  x = x0
  @test unscale_objective(nlp_scaled, obj(nlp_scaled, x)) ≈ obj(nlp, x)
  @test unscale_constraints(nlp_scaled, cons(nlp_scaled, x)) ≈ cons(nlp, x)

  y = ones(2)
  y_scaled = scale_multipliers(nlp_scaled, y)
  @test unscale_multipliers(nlp_scaled, y_scaled) ≈ y

  # No-op fallbacks on a model that isn't scaled
  @test unscale_objective(nlp, 1.0) == 1.0
  @test unscale_constraints(nlp, [1.0, 2.0]) == [1.0, 2.0]
  @test unscale_multipliers(nlp, [1.0, 2.0]) == [1.0, 2.0]
  @test scale_multipliers(nlp, [1.0, 2.0]) == [1.0, 2.0]

  # update_scaling! recomputes d_f and d_c from a gradient and Jacobian
  gk = grad(nlp, x)
  Ak = jac(nlp, x)
  gmax = 10.0
  update_scaling!(nlp_scaled, gk, Ak; gmax = gmax)

  @test nlp_scaled.d_f == gmax / norm(gk, Inf)
  for j in axes(Ak, 1)
    nc = norm(view(Ak, j, :), Inf)
    @test nlp_scaled.d_c[j] == (nc > 0 ? min(1.0, gmax / nc) : 1.0)
  end

  # find_model unwraps to a ScaledModel, or returns nothing if there is none
  @test find_model(ScaledModel, nlp_scaled) === nlp_scaled
  @test find_model(ScaledModel, nlp) === nothing
end
