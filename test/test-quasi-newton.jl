@testset "Compact BFGS" begin
  nlp = CUTEstModel("BT1")

  # Test pushing without scaling
  n = nlp.meta.nvar
  compact_bfgs_nlp =
    CompactBFGSModel(nlp, mem = 5, scaling = false, max_skip = typemax(Int))
  bfgs_op_nlp = LBFGSModel(nlp, scaling = false)

  for _ = 1:100
    s = randn(n)
    y = randn(n)
    push!(compact_bfgs_nlp, s, y)
    push!(bfgs_op_nlp, s, y)
  end

  Uk, Vk = compact_bfgs_nlp.op.Uk, compact_bfgs_nlp.op.Vk

  B0 = compact_bfgs_nlp.op.ξ*I(n)
  Bk_compact = B0 - Uk*Uk' + Vk*Vk'
  Bk_op = Matrix(bfgs_op_nlp.op)

  @test norm(Bk_compact - Bk_op) / norm(Bk_op) <= 1e-12

  s = randn(n)
  y = randn(n)
  nallocs = @allocated push!(compact_bfgs_nlp, s, y)
  @test nallocs == 0

  compact_bfgs_nlp = CompactBFGSModel(nlp, mem = 5, max_skip = typemax(Int))
  bfgs_op_nlp = LBFGSModel(nlp)

  for _ = 1:100
    s = randn(n)
    y = randn(n)
    push!(compact_bfgs_nlp, s, y)
    push!(bfgs_op_nlp, s, y)
  end

  Uk, Vk = compact_bfgs_nlp.op.Uk, compact_bfgs_nlp.op.Vk

  B0 = compact_bfgs_nlp.op.ξ*I(n)
  Bk_compact = B0 - Uk*Uk' + Vk*Vk'
  Bk_op = Matrix(bfgs_op_nlp.op)

  @test norm(Bk_compact - Bk_op) / norm(Bk_op) <= 1e-12

  s = randn(n)
  y = randn(n)
  nallocs = @allocated push!(compact_bfgs_nlp, s, y)
  @test nallocs == 0

  finalize(nlp)
end
