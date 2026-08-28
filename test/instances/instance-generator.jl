"""
Generate a (dense) random instance of the ℓ₂ penalty algorithm subproblem: 
```math
min_u \tfrac{1}{2} u^T Q u + g^T u + \tau | Au + b |_2 
```

Arguments:
* `n, m`: sizes of Q and A
* `alpha`: scalar >= 0
* `κ_Q`: condition number of Q
* `κ_A`: condition number of A
* `λmin`: minimal eigenvalue of Q (scales spectrum)
* `σmin`: minimal singular value of A (scales spectrum)
* `mode_Q, mode_A`: :exact or :approx (spectrum exact or approximate)
"""
function generate_instance(
  n::Int,
  m::Int,
  alpha::Real;
  κ_Q::Real = 10.0,
  λmin::Real = 1.0,
  κ_J::Real = 10.0,
  σmin::Real = 1.0,
  rng = Xoshiro(123),
  Hessian_modifier = H -> H,
)

  @assert κ_Q >= 1 && κ_J >= 1 "Condition numbers must be >= 1"
  @assert λmin > 0 && σmin > 0 "Minimal eigen/singular values must be > 0"
  @assert alpha >= 0 "alpha must be nonnegative"

  # --- Generate Q ---
  λmax = λmin * κ_Q
  eigsQ = range(λmin, λmax, length = n) |> collect
  UQ = qr(randn(rng, n, n)).Q
  Q = UQ * Diagonal(eigsQ) * UQ'
  σ = σmin

  # --- Generate J ---
  σmax = σmin * κ_J
  svalsJ = range(σmin, σmax, length = m) |> collect
  UJ = qr(randn(rng, m, m)).Q
  VJ = qr(randn(rng, n, n)).Q
  Σ = zeros(m, n)
  Σ[1:m, 1:m] .= Diagonal(svalsJ)
  J = UJ * Σ * VJ'

  # --- Generate vectors ---
  nabla = randn(rng, n)
  b = randn(rng, m)

  # --- Solve KKT ---
  KKT = [Q+σ*I J'; J -alpha * I(m)]
  rhs = vcat(-nabla, -b)
  sol = KKT \ rhs
  u = sol[1:n]
  y = sol[(n+1):end]
  tau = norm(y)

  if alpha == 0
    tau += abs(randn(rng)) + 1e-6
  end

  x0 = zeros(typeof(alpha), n)
  H =
    Hessian_modifier == LinearOperator ?
    Hessian_modifier{Float64,Vector{Float64}}(Q, symmetric = true) : Hessian_modifier(Q)
  model = QuadraticModel(nabla, H, regularize = true, σ = σ, x0 = x0)
  c! = let c = b
    (b, x) -> begin
      b .= c
    end
  end
  J! = let J = J
    (A, x) -> begin
      A .= J
    end
  end
  h = ShiftedCompositeNormL2(tau, c!, J!, SparseMatrixCOO(J), b)
  return ShiftedL2PenalizedProblem(
    model,
    h,
    nothing,
    model.meta,
    nothing,
    nothing,
    true,
  ),
  Dict(:u => u, :y => y, :tau => tau)
end
