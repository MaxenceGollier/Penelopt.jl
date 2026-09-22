@testset "LogBarrierModel" begin
  μ = 0.1
  l = [-Inf, 0.0, 0.1]
  u = [1.0, Inf, 2.0]
  x = [0.2, 0.5, 0.7]
  y = [0.3]

  f(x) = sum(abs2, x)
  c(x) = [sum(x .^ 3) - 1]
  ϕ(x) = -μ * (log(1 - x[1]) + log(x[2]) + log(x[3] - 0.1) + log(2 - x[3]))

  nlp = ADNLPModel(f, x, l, u, c, [0.0], [0.0])
  ref = ADNLPModel(x -> f(x) + ϕ(x), x, c, [0.0], [0.0])
  bnlp = LogBarrierModel(nlp; μ)

  @test all(==(-Inf), bnlp.meta.lvar) && all(==(Inf), bnlp.meta.uvar)
  @test obj(bnlp, x) ≈ obj(ref, x)
  @test cons(bnlp, x) ≈ cons(ref, x)

  # The barrier multipliers z_l = μ/(x-l), z_u = μ/(u-x) are not computed by the
  # NLPModels API; they must be updated at x before evaluating derivatives.
  Penelopt.update_multipliers!(bnlp.ϕ, x)
  @test bnlp.ϕ.z_l ≈ [0.0, μ / 0.5, μ / 0.6]
  @test bnlp.ϕ.z_u ≈ [μ / 0.8, 0.0, μ / 1.3]
  @test grad(bnlp, x) ≈ grad(ref, x)
  @test Matrix(hess(bnlp, x, y)) ≈ Matrix(hess(ref, x, y))
  @test Matrix(hess(bnlp, x, y, obj_weight = 2.0)) ≈ Matrix(hess(ref, x, y, obj_weight = 2.0))
  @test obj(bnlp, [1.5, 0.5, 0.7]) == Inf
  @test Penelopt.isinterior(bnlp.ϕ, x) && !Penelopt.isinterior(bnlp.ϕ, [1.0, 0.5, 0.7])

  @test Penelopt.get_model(bnlp) === nlp
  @test add_log_barrier(bnlp; μ) === bnlp
  @test add_log_barrier(nlp; μ) isa LogBarrierModel

  pb = BarrierPenalizedProblem(nlp; μ)
  set_barrier!(pb, 1.0)
  @test pb.model.ϕ.μ == 1.0

  @test_throws AssertionError Penelopt.LogBarrier(-1.0, l, u)
  @test_throws AssertionError Penelopt.LogBarrier(1.0, u, l)
end
