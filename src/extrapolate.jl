function extrapolate!(
  nlp::AbstractNLPModel,
  x::V,
  solver::L2PenaltySolver{T,V,S,PB},
  τ₂::T,
  τ₁::T,
) where {T,V,S,N<:NullHessianModel{T,V},PB<:L2PenalizedProblem{T,V,N}}
  return false
end

function extrapolate!(
  x::V,
  solver::L2PenaltySolver{T,V,S,PB},
  τ₂::T,
  τ₁::T,
) where {T,V,S,N<:AbstractNLPModel{T,V},PB<:L2PenalizedProblem{T,V,N}}

  # Retrieve workspace
  subsolver, substats = solver.subsolver, solver.substats
  ms_solver, ms_stats = subsolver.subsolver, subsolver.substats
  mk = subsolver.subpb
  φ, ψ = mk.model, mk.h
  nlp, h = mk.parent.model, mk.parent.h
  fk, hk = substats.solver_specific[:smooth_obj], substats.solver_specific[:nonsmooth_obj]
  xk, xkn, s, y = subsolver.xk, subsolver.xkn, subsolver.s, subsolver.y
  ∇fk = φ.data.c
  n, m = nlp.meta.nvar, nlp.meta.ncon
  α = ms_stats.solver_specific[:alpha]

  # (x, y) is an (approximate) solution of 
  # min_x f(x) + τ₁ ‖ c(x) ‖₂ = min_x max_y f(x) +  yᵀ c(x) s.t. ‖y‖₂ ≤ τ₁ 
  # If ‖y‖₂ < τ₁, then no extrapolation is needed.
  # Else, we consider that τ₁ = ‖y‖₂.
  norm_y = norm(y, 2)
  norm_y < τ₁ && return false
  τ₁ = norm_y

  update_workspace!(ms_solver.workspace, φ.data.H, ψ.A, zero(T), α)

  # [ H + σI Aᵀ][px] = -[0]
  # [   A    0 ][py] = -[y] 
  @views @. ms_solver.u2[(n+1):(n+m)] = -ms_solver.x1[(n+1):(n+m)]
  solve_system!(ms_solver.workspace, ms_solver.u2)
  get_solution!(ms_solver.x2, ms_solver.workspace)
  @views px, py = ms_solver.x2[1:n], ms_solver.x2[(n+1):(n+m)]

  # Check inertia and safeguard the solution.
  npos, nzero, nneg = get_inertia(ms_solver.workspace)
  status = get_status(ms_solver.workspace)
  if status == :failed || npos != n || nneg != m || nzero != 0
    return false
  end

  # α' = ‖y‖₂ / (yᵀ y')
  α_dot = norm_y/dot(y, py)

  # x' = α' p_x
  px .*= α_dot

  # x = x + (τ₂ - τ₁) x'
  x_extrap = px
  x_extrap .= x .+ (τ₂ - τ₁) .* px

  # Basic safeguard
  obj_extrap = obj(nlp, x_extrap)
  cons!(nlp, x_extrap, solver.cn)

  if obj_extrap < Inf && norm(solver.cn) < Inf
    solver.x .= x_extrap
    set_solver_specific!(solver.substats, :smooth_obj, obj_extrap)
    return true
  end

  return false
end
