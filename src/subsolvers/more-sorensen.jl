export MoreSorensenSolver
import Base.show

mutable struct MoreSorensenSolver{
  T<:Real,
  V<:AbstractVector{T},
  L<:Union{AbstractLinearOperator,AbstractMatrix},
  W,
} <: AbstractPenalizedProblemSolver
  u1::V
  u2::V
  x1::V
  x2::V
  H::L
  workspace::W
end

function MoreSorensenSolver(
  reg_nlp::AbstractRegularizedNLPModel{T,V};
  solver = :ldlt,
) where {T,V}
  x0 = reg_nlp.model.meta.x0
  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  u1 = similar(x0, n+m)
  u2 = zeros(eltype(x0), n+m)
  x1 = zeros(eltype(x0), n+m)
  x2 = zeros(eltype(x0), n+m)

  # Check linear solver

  # Check for MUMPS
  mumps_loaded = !isnothing(Base.get_extension(@__MODULE__, :ExactPenaltyMUMPSExt))
  if !mumps_loaded && solver == :mumps
    warning("ExactPenalty.jl: MUMPS extension is not loaded. Please install MPI.jl and MUMPS.jl. Switching to LDLFactorizations.jl...")
    solver = :ldlt
  end

  # Check for HSL
  hsl_loaded = !isnothing(Base.get_extension(@__MODULE__, :ExactPenaltyHSLExt))
  if !hsl_loaded && solver == :ma57
    warning("ExactPenalty.jl: HSL extension is not loaded. Please install HSL.jl. Switching to LDLFactorizations.jl...")
    solver = :ldlt
  end

  hsl_isfunctional = hsl_loaded && hsl_functional()
  if !hsl_isfunctional && solver == :ma57
    warning("ExactPenalty.jl: HSL extension is not functional. Please check your license and make sure you have loaded HSL_jll.jl appropriately. Switching to LDLFactorizations.jl...")
    solver = :ldlt
  end

  # Check for Krylov
  krylov_loaded = !isnothing(Base.get_extension(@__MODULE__, :ExactPenaltyKrylovExt))
  if !krylov_loaded && solver == :minres_qlp
    warning("ExactPenalty.jl: Krylov extension is not loaded. Please install Krylov.jl. Switching to LDLFactorizations.jl...")
    solver = :ldlt
  end

  H = K2(
    n,
    m,
    n+m,
    n+m,
    zero(T),
    reg_nlp.model.data.σ,
    reg_nlp.h.A,
    reg_nlp.model.data.H;
    format = solver ∈ [:ma57, :mumps] ? :coo : :csc,
    int_type = solver == :mumps ? Int32 : Int64,
  )
  workspace = construct_workspace(H, u1, n, m; solver = solver)

  return MoreSorensenSolver(u1, u2, x1, x2, H, workspace)
end

function SolverCore.solve!( #TODO add verbose and kwargs
  solver::MoreSorensenSolver{T,V},
  reg_nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  stats::GenericExecutionStats{T,V,V};
  x = reg_nlp.model.meta.x0,
  print_level::Int = 0,
  verbose::Int = 1,
  atol::T = eps(T)^(0.6),
  max_time::T = T(30),
  max_iter::Int = 10,
  μα::T = T(0.1),
  μσ::T = T(10),
  α0::T = eps(T),
  αmin1::T = eps(T)^(0.8),
  αmin2::T = eps(T)^(0.6),
  σmax::T = 1 / eps(T),
  accept_descent::Bool = true, # Whether we accept inexact steps that decrease the quadratic model.
) where {T,V,M,H,P}
  start_time = time()
  set_time!(stats, 0.0)
  set_iter!(stats, 0)

  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  Δ = reg_nlp.h.h.lambda

  u1, u2, x1, x2 = solver.u1, solver.u2, solver.x1, solver.x2
  solver_workspace = solver.workspace

  # Create problem
  @. u1[1:n] = -reg_nlp.model.data.c
  @. u1[(n+1):(n+m)] = -reg_nlp.h.b

  α = α0
  update_workspace!(
    solver_workspace,
    reg_nlp.model.data.H,
    reg_nlp.h.A,
    reg_nlp.model.data.σ,
    α,
  )

  if print_level > 0
    @info introduction_message(solver, Δ)
    @info separator(type = :ms_loop)
    @info header_message(type = :ms_loop)
    @info separator(type = :ms_loop)
  end

  αmin = αmin1

  # [ H + σI Aᵀ][x] = -[∇f]
  # [   A    0 ][y] = -[c] 
  solve_system!(solver_workspace, u1)
  get_solution!(x1, solver_workspace)
  npos, nzero, nneg = get_inertia(solver_workspace)
  status = get_status(solver_workspace)

  # Get correct inertia
  # If the factorization/solver failed, it in indicates we should add a minimal regularization too.
  if nneg < m || status == :failed
    α = αmin
    set_dual_inertia!(solver_workspace, α)
    solve_system!(solver_workspace, u1)
    get_solution!(x1, solver_workspace)
    npos, nzero, nneg = get_inertia(solver_workspace)
    status = get_status(solver_workspace)

    if nneg < m || status == :failed
      αmin = αmin2
      α = αmin
      set_dual_inertia!(solver_workspace, α)
      solve_system!(solver_workspace, u1)
      get_solution!(x1, solver_workspace)
      npos, nzero, nneg = get_inertia(solver_workspace)
      status = get_status(solver_workspace)
    end
  end

  while (npos < n || status == :failed) && reg_nlp.model.data.σ <= σmax

    reg_nlp.model.data.σ *= μσ
    set_primal_inertia!(solver_workspace, reg_nlp.model.data.σ)

    # [ H + σI Aᵀ][x] = -[∇f]
    # [   A    0 ][y] = -[c] 
    solve_system!(solver_workspace, u1)
    get_solution!(x1, solver_workspace)
    npos, nzero, nneg = get_inertia(solver_workspace)
    status = get_status(solver_workspace)
  end

  if reg_nlp.model.data.σ >= σmax
    set_status!(stats, :exception)
    return
  end

  is_descent = check_descent(reg_nlp, @view x1[1:n])
  norm_x1 = norm(@view x1[(n+1):(n+m)])

  if print_level > 0 && stats.iter % verbose == 0
    @info log_ms_iteration(stats, reg_nlp.model.data.σ, α, norm_x1, Δ, npos, nzero, nneg, status, is_descent)
  end

  if norm_x1 <= Δ || (is_descent && accept_descent)
    set_solution!(stats, @view x1[1:n])
    set_status!(stats, :first_order)

    !is_descent && set_status!(stats, :not_desc)
    set_solver_specific!(stats, :alpha, α)
    print_level > 0 && @info conclusion_message(solver, stats)

    return
  end

  # [ H + σI Aᵀ][x'] = -[0]
  # [   A    0 ][y'] = -[x] 
  @views @. u2[(n+1):(n+m)] = -x1[(n+1):(n+m)]
  solve_system!(solver_workspace, u2)
  get_solution!(x2, solver_workspace)

  while abs(norm_x1 - Δ) > atol && stats.iter < max_iter && stats.elapsed_time < max_time
    # α = α + (‖y‖/Δ - 1)*‖y‖²/(yᵀy')
    @views α₊ = α + norm_x1^2/dot(x1[(n+1):(n+m)], x2[(n+1):(n+m)])*(norm_x1/Δ - 1)

    α = α₊ ≤ 0 ? max(μα*α, αmin) : α₊
    set_dual_inertia!(solver_workspace, α)

    # [ H + σI  Aᵀ ][x] = -[∇f]
    # [   A    -αI ][y] = -[c] 
    solve_system!(solver_workspace, u1)
    get_solution!(x1, solver_workspace)

    # Check whether x1 decreases the model.
    is_descent = check_descent(reg_nlp, @view x1[1:n])
    norm_x1 = norm(@view x1[(n+1):(n+m)])

    if is_descent && accept_descent
      set_solution!(stats, @view x1[1:n])
      set_status!(stats, :first_order)
      set_solver_specific!(stats, :alpha, α)
      set_iter!(stats, stats.iter + 1)
      if print_level > 0 && stats.iter % verbose == 0
        @info log_ms_iteration(stats, reg_nlp.model.data.σ, α, norm_x1, Δ, npos, nzero, nneg, status, is_descent)
      end
      print_level > 0 && @info conclusion_message(solver, stats)
      return
    end

    # Check whether the matrix still has the correct inertia. (We may have failed to detect earlier)
    npos, nzero, nneg = get_inertia(solver_workspace)
    if npos < n
      reg_nlp.model.data.σ *= μσ
      if reg_nlp.model.data.σ >= σmax
        set_status!(stats, :exception)
        print_level > 0 && @info conclusion_message(solver, stats)
        return
      end
      solve!(solver, reg_nlp, stats)
    end

    # [ H + σI  Aᵀ ][x'] = -[0]
    # [   A    -αI ][y'] = -[x]
    @views @. u2[(n+1):(n+m)] = -x1[(n+1):(n+m)]
    solve_system!(solver_workspace, u2)
    get_solution!(x2, solver_workspace)

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time()-start_time)

    if print_level > 0 && stats.iter % verbose == 0
      @info log_ms_iteration(stats, reg_nlp.model.data.σ, α, norm_x1, Δ, npos, nzero, nneg, status, is_descent)
    end

    α == αmin && break
  end

  set_solution!(stats, @view x1[1:n])
  set_status!(stats, :first_order)
  set_solver_specific!(stats, :alpha, α)

  stats.iter >= max_iter && set_status!(stats, :max_iter)
  stats.elapsed_time >= max_time && set_status!(stats, :max_time)
  !check_descent(reg_nlp, @view x1[1:n]) && set_status!(stats, :not_desc)
  if !check_descent(reg_nlp, @view x1[1:n])
    reg_nlp.model.data.σ *= μσ
    if reg_nlp.model.data.σ >= σmax
      set_status!(stats, :not_desc)
      print_level > 0 && @info conclusion_message(solver, stats)
      return
    end
    solve!(solver, reg_nlp, stats)
  end
end

function SolverCore.solve!(
  solver::MoreSorensenSolver{T,V},
  reg_nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  stats::GenericExecutionStats{T,V,V};
  x = reg_nlp.model.meta.x0,
  print_level::Int = 0,
  verbose::Int = 1,
  atol::T = eps(T)^(0.6),
  max_time::T = T(30),
  max_iter::Int = 10,
  μα::T = T(0.1),
  μσ::T = T(10),
  α0::T = eps(T),
  αmin1::T = eps(T)^(0.8),
  αmin2::T = eps(T)^(0.6),
  σmax::T = 1 / eps(T),
  accept_descent::Bool = true, # Whether we accept inexact steps that decrease the quadratic model.
) where {T,V,M,H,O<:NullHessianModel,P<:L2PenalizedProblem{T,V,O}}

  n = reg_nlp.model.meta.nvar
  ψ = reg_nlp.h
  u1, x1 = solver.u1, solver.x1

  ν = 1 / reg_nlp.model.data.σ
  @. u1[1:n] = - ν * reg_nlp.model.data.c

  @views prox!(
    x1[1:n],
    ψ,
    u1[1:n],
    ν,
    max_iter = max_iter,
    max_time = max_time,
    atol = atol,
  )

  @. x1[(n+1):end] = - ψ.q / ν
  set_solution!(stats, @view x1[1:n])
  set_status!(stats, :first_order)
  !check_descent(reg_nlp, @view x1[1:n]) && set_status!(stats, :not_desc)
end

function get_primal_dual_sol!(s, y, solver::MoreSorensenSolver)
  n = length(s)
  s .= @view solver.x1[1:n]
  y .= @view solver.x1[(n+1):end]
end

function SolverCore.reset!(solver::MoreSorensenSolver{T}) where {T}
  set_n_fact!(solver.workspace, 0)
end
