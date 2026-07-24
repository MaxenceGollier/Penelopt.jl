export L2Penalty, L2PenaltySolver, solve!

import SolverCore.solve!

mutable struct L2PenaltySolver{
  T<:Real,
  V<:AbstractVector{T},
  S<:AbstractOptimizationSolver,
  PB<:AbstractRegularizedNLPModel,
} <: AbstractOptimizationSolver
  x::V
  xn::V
  y::V
  cn::V
  dual_res::V
  s::V
  s0::V
  ∇fk::V
  temp_b::V
  subsolver::S
  subpb::PB
  substats::GenericExecutionStats{T,V,V,T}
end

function L2PenaltySolver(
  nlp::AbstractNLPModel{T,V};
  r2n_m_monotone::Int=12,
  linear_solver::String="ldlt",
) where {T,V}
  x0 = nlp.meta.x0
  x, xn, s, s0 = similar(x0), similar(x0), similar(x0), zero(x0)
  temp_b = similar(x0, nlp.meta.ncon)
  dual_res = similar(x0)
  cn = similar(x0, nlp.meta.ncon)
  y = similar(x0, nlp.meta.ncon)
  ∇fk = similar(x0)

  penalty_subproblem = L2PenalizedProblem(nlp) # f(x) + τ‖c(x)‖₂
  substats = GenericExecutionStats(penalty_subproblem, solver_specific = Dict{Symbol,T}())
  solver = PenaltyR2NSolver(penalty_subproblem; m_monotone = r2n_m_monotone, linear_solver = linear_solver)

  set_solver_specific!(substats, :primal_ktol, T(0))
  set_solver_specific!(substats, :dual_ktol, T(0))
  set_solver_specific!(substats, :n_fact, T(0))
  set_solver_specific!(substats, :tau, T(0))
  set_solver_specific!(substats, :sigma, T(0))
  set_solver_specific!(substats, :rho, T(0))

  return L2PenaltySolver(
    x,
    xn,
    y,
    cn,
    dual_res,
    s,
    s0,
    ∇fk,
    temp_b,
    solver,
    penalty_subproblem,
    substats,
  )
end

function SolverCore.reset!(solver::L2PenaltySolver)
  SolverCore.reset!(solver.subsolver)
end

include("logging.jl")

"""
    L2Penalty(nlp; kwargs…)

An exact ℓ₂-penalty method for the problem

    min f(x) 	s.t c(x) = 0

where f: ℝⁿ → ℝ and c: ℝⁿ → ℝᵐ respectively have a Lipschitz-continuous gradient and Jacobian.

At each iteration k, an iterate is computed as 

    xₖ ∈ argmin f(x) + τₖ‖c(x)‖₂

where τₖ is some penalty parameter.
This nonsmooth problem is solved using `R2` (see `R2` for more information) with the first order model ψ(s;x) = τₖ‖c(x) + J(x)s‖₂

For advanced usage, first define a solver "L2PenaltySolver" to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = L2PenaltySolver(nlp)
    solve!(solver, nlp)

    stats = ExactPenaltyExecutionStats(nlp)
    solver = L2PenaltySolver(nlp)
    solve!(solver, nlp, stats)

# Arguments
* `nlp::AbstractNLPModel{T, V}`: the problem to solve, see `RegularizedProblems.jl`, `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess;
- `atol::T = √eps(T)`: absolute tolerance;
- `rtol::T = √eps(T)`: relative tolerance;
- `sub_atol::T = zero(T)`: absolute tolerance given to the subsolver;
- `sub_rtol::T = T(1e-2)`: relative tolerance given to the subsolver;
- `infeasible_tol = T(1e-2)`: tolerance used to decide whether the problem is infeasible or not √θₖ/‖c(xₖ)‖₂ < infeasible_tol, the problem is declared infeasible.
- `max_eval::Int = -1`: maximum number of evaluation of the objective function (negative number means unlimited);
- `sub_max_eval::Int = -1`: maximum number of evaluation for the subsolver (negative number means unlimited);
- `max_time::Float64 = 30.0`: maximum time limit in seconds;
- `max_iter::Int = 10000`: maximum number of iterations;
- `sub_max_iter::Int = 10000`: maximum number of iterations for the subsolver;
- `max_decreas_iter::Int = 10`: maximum number of iteration where ‖c(xₖ)‖₂ does not decrease before calling the problem locally infeasible;
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration;
- `sub_verbose::Int = 0`: if > 0, display subsolver iteration details every `verbose` iteration;
- `τ::T = T(100)`: initial penalty parameter;
- `β1::T = T(1)`: minimal penalty parameter increase,
- `β3::T = 1/τ`: initial regularization parameter σ₀ = β3/τₖ at each iteration;
- `β4::T = eps(T)`: minimal regularization parameter σ for `R2`;
- `primal_feasibility_mode::Symbol = :kkt`: describes how the primal feasibility is computed during the outer iterations. 
                                            With `:kkt`, the primal feasibility is the infinity norm of the residual ‖c(xₖ)‖∞.
                                            With `:decrease`, the primal feasibility is computed as a model decrease of the feasibility problem.
- `dual_feasibility_mode::Symbol = :kkt`: describes how the dual feasibility is computed during the outer and inner iterations. 
                                          With `:kkt`, the dual feasibility is the infinity norm of the residual ‖∇fₖ + Jₖᵀyₖ‖∞, where yₖ 
                                          is resulting from the computation of the Cauchy point of the subproblem.  
                                          With `:decrease`, the dual feasibility is computed as a model decrease with respect to the Cauchy point. 

other 'kwargs' are passed to `R2` (see `R2` for more information).

The algorithm stops either when `√θₖ < atol + rtol*√θ₀ ` or `θₖ < 0` and `√(-θₖ) < neg_tol` where θₖ := ‖c(xₖ)‖₂ - ‖c(xₖ) + J(xₖ)sₖ‖₂, and √θₖ is a stationarity measure.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.subsolver`: a `PenaltyR2Solver` structure holding relevant information on the subsolver state, see `R2` for more information;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.
You can also use the `sub_callback` keyword argument which has exactly the same structure and in sent to `R2`.
"""
function L2Penalty(
  nlp::AbstractNLPModel{T,V};
  r2n_m_monotone::Int=12,
  linear_solver::String="ldlt",
  kwargs...
) where {T<:Real,V}

  if !equality_constrained(nlp)
    error("L2Penalty: This algorithm only works for equality contrained problems.")
  end

  solver = L2PenaltySolver(
    nlp;
    r2n_m_monotone=r2n_m_monotone,
    linear_solver=linear_solver,
  )

  stats = ExactPenaltyExecutionStats(nlp)
  solve!(solver, nlp, stats; kwargs...)
  return stats
end

function SolverCore.solve!(
  solver::L2PenaltySolver{T,V},
  nlp::AbstractNLPModel{T,V},
  stats::GenericExecutionStats{T,V,V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,

  ## Termination arguments
  atol::T = √eps(T),
  rtol::T = √eps(T),
  dual_inf_atol::T = zero(T),
  dual_inf_rtol::T = zero(T),
  primal_inf_atol::T = zero(T),
  primal_inf_rtol::T = zero(T),
  max_eval::Int = -1,
  max_time::Float64 = 30.0,
  max_iter::Int = 100,
  r2n_max_iter::Int = 1000,
  ms_max_iter::Int = 10,
  μ::T = T(1e-2),
  infeasible_tol::T = T(1e-2),
  infeasible_iter::Int = 3,
  
  ## Logging arguments
  print_level::Int = 0,
  verbose::Int = 1,
  r2n_verbose::Int = 1,
  ms_verbose::Int = 1,

  ## R2N Specific arguments
  r2n_η1::T = √√eps(T),
  r2n_η2::T = isa(nlp, QuasiNewtonModel) ? T(0.9) : T(0.1),
  r2n_γ::T = T(3),
  r2n_watchdog_max_iter::Int = 10,
  r2n_watchdog_η0::T = √eps(T),
  r2n_tiny_step_tol::T = eps(T),

  ## MS Specific arguments
  ms_accept_descent::Bool = true,
  ms_σmax::T = 1/eps(T),
  ms_tol::T = eps(T)^(0.6),
  ms_μα::T = T(0.1),
  ms_μσ::T = T(10),
  ms_α0::T = eps(T),
  ms_αmin1::T = isa(nlp, QuasiNewtonModel) ? eps(T)^(0.6) : eps(T)^(0.8),
  ms_αmin2::T = eps(T)^(0.6),

  ## Other arguments
  max_decreas_iter::Int = 10,
  τ::T = T(100),
  β1::T = T(1),
  β3::T = 1e-4/τ,
  β4::T = eps(T),
) where {T,V}
  reset!(stats)

  # Retrieve workspace
  penalty_pb = solver.subpb # f(x) + τ‖c(x)‖₂
  mk = solver.subsolver.subpb
  φ, ψ = mk.model, mk.h

  x = solver.x .= x
  y = solver.y

  shift!(ψ, x)
  fx = obj(nlp, x)
  hx = norm(ψ.b)

  set_iter!(stats, 0)
  rem_eval = max_eval
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fx)

  ## Compute Feasibility

  primal_feas = kkt_primal_feas!(solver)

  set_solver_specific!(solver.substats, :smooth_obj, fx)
  grad!(nlp, x, solver.∇fk)
  compute_least_square_multipliers!(solver)
  dual_feas = least_square_dual_feas!(solver)
  solver.subsolver.y .= solver.y

  primal_tol = max(primal_inf_atol, atol) + max(primal_inf_rtol, rtol) * primal_feas
  dual_tol = max(dual_inf_atol, atol) + max(dual_inf_rtol, rtol) * dual_feas

  primal_ktol = one(primal_tol)
  dual_ktol = min(one(dual_tol), max(μ * dual_feas, dual_tol))
  dual_krtol = T(0)

  set_solver_specific!(solver.substats, :primal_ktol, primal_ktol)
  set_solver_specific!(solver.substats, :dual_ktol, dual_ktol)
  set_residuals!(stats, dual_feas, primal_feas)

  solved = dual_feas ≤ dual_tol && primal_feas ≤ primal_tol

  ## Initialize penalty parameter
  τ = max(norm(solver.y, 1), T(1))
  set_penalty!(mk, τ)
  νsub = 1 / β4
  set_solver_specific!(solver.substats, :tau, τ)

  ## Logging
  if print_level > 0
    @info introduction_message(solver, nlp)
    @info separator()
    @info header_message()
    @info separator()
    @info log_iteration(solver, nlp, stats)
  end

  ## Initialize Model
  shift!(mk, x, ∇f = solver.∇fk, y = y)

  infeasible = false
  not_desc = false
  n_iter_since_decrease = 0
  primal_decrease = false
  first_increase = true

  set_status!(
    stats,
    get_status(
      nlp,
      elapsed_time = stats.elapsed_time,
      n_iter_since_decrease = n_iter_since_decrease,
      iter = stats.iter,
      optimal = solved,
      infeasible = infeasible,
      not_desc = not_desc,
      max_eval = max_eval,
      max_time = max_time,
      max_iter = max_iter,
      max_decreas_iter = max_decreas_iter,
    ),
  )

  callback(nlp, solver, stats)

  done = stats.status != :unknown

  while !done

    solve!(
      solver.subsolver,
      solver.subpb,
      solver.substats;
      x = x,
      atol = dual_ktol,
      rtol = dual_krtol,
      print_level = print_level - 1,
      verbose = r2n_verbose,
      max_iter = r2n_max_iter,
      ms_max_iter = ms_max_iter,
      max_time = max_time - stats.elapsed_time,
      max_eval = rem_eval,
      σmin = β4,
      σk = 1 / νsub,
      η1 = r2n_η1,
      η2 = r2n_η2,
      γ = r2n_γ,
      watchdog_max_iter = r2n_watchdog_max_iter,
      watchdog_η0 = r2n_watchdog_η0,
      tiny_step_tol = r2n_tiny_step_tol,
      is_shifted = true,
      primal_decrease = primal_decrease,
      first_increase = first_increase,
    )

    if solver.substats.status == :unbounded
      τ *= 10
      set_penalty!(mk, τ)
      νsub = 1 / β4
      shift!(mk, x, y = y)
      set_solver_specific!(solver.substats, :smooth_obj, fx)
      continue
    end

    if solver.substats.status == :not_desc
      not_desc = true
    end

    x .= solver.substats.solution
    y .= solver.subsolver.y
    fx = solver.substats.solver_specific[:smooth_obj]
    hx_prev = copy(hx)
    hx = solver.substats.solver_specific[:nonsmooth_obj]/τ
    solver.∇fk .= φ.data.c

    ## Compute feasibility 
    primal_feas = kkt_primal_feas!(solver)
    dual_feas = kkt_dual_feas!(solver)

    if primal_feas > primal_ktol || (dual_ktol ≤ dual_tol && primal_feas > primal_tol)
      # Update penalty parameter
      τ₊ = max(τ + β1, norm(y, 1))
      if extrapolate!(x, solver, τ₊, τ)
        shift!(mk, x, y = y)
        set_solver_specific!(solver.substats, :smooth_obj, obj(nlp, x))

        # Subsolver: Do not impose primal decrease
        primal_decrease = false
      else

        # Subsolver: Impose primal decrease
        primal_decrease = true
      end
      τ = τ₊
      set_penalty!(mk, τ)

      # Initialize regularization parameter
      νsub = 1 / solver.substats.solver_specific[:sigma]

      # Subsolver: Activate the aggressive regularization parameter update if sigma is too small
      first_increase = true

      # Add a relative tolerance for the subsolver
      dual_ktol = dual_tol
      set_solver_specific!(solver.substats, :dual_ktol, dual_ktol)
      set_solver_specific!(solver.substats, :tau, τ)
      dual_krtol = μ
    else
      # Tighten tolerances
      primal_ktol = max(μ*primal_feas, primal_tol)
      dual_ktol = max(μ*dual_feas, dual_tol)
      dual_krtol = T(0)
      set_solver_specific!(solver.substats, :primal_ktol, primal_ktol)
      set_solver_specific!(solver.substats, :dual_ktol, dual_ktol)

      y .= solver.substats.multipliers

      # Initialize regularization parameter
      νsub = 1/solver.substats.solver_specific[:sigma]

      # Subsolver: Do not impose primal decrease
      primal_decrease = false

      # Subsolver: Desactivate the aggressive regularization parameter update if sigma is too small
      first_increase = false
    end

    # Check whether the primal feasibility has decreased. If not, increase the penalty parameter more aggressively.
    if primal_feas > primal_ktol && hx_prev < hx
      n_iter_since_decrease += 1
      β1 *= 10
    else
      n_iter_since_decrease = 0
    end

    solved = dual_feas ≤ dual_tol && primal_feas ≤ primal_tol

    # Infeasiblity detection
    if stats.iter % infeasible_iter == 0
      θ = compute_θ!(solver)

      infeasible =
        hx > primal_tol &&
        sqrt(max(θ, 0))/hx < infeasible_tol &&
        sqrt(max(θ, 0)) < primal_ktol
    end

    set_iter!(stats, stats.iter + 1)
    rem_eval = max_eval - neval_obj(nlp)
    set_time!(stats, time() - start_time)
    set_objective!(stats, fx)
    set_residuals!(stats, primal_feas, dual_feas)
    set_constraint_multipliers!(stats, solver.y)
    set_solver_specific!(stats, :n_fact, solver.substats.solver_specific[:n_fact])

    set_status!(
      stats,
      get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        n_iter_since_decrease = n_iter_since_decrease,
        iter = stats.iter,
        optimal = solved,
        infeasible = infeasible,
        not_desc = not_desc,
        small_step = solver.substats.status == :small_step,
        max_eval = max_eval,
        max_time = max_time,
        max_iter = max_iter,
        max_decreas_iter = max_decreas_iter,
      ),
    )

    ## Log status
    if print_level > 0 && stats.iter % verbose == 0
      if stats.iter % (20 * verbose) == 0 && stats.iter > 0
        @info separator()
        @info header_message()
        @info separator()
      end
      @info log_iteration(solver, nlp, stats)
    end

    callback(nlp, solver, stats)

    done = stats.status != :unknown
  end

  if print_level > 0
    @info conclusion_message(solver, nlp, stats)
  end

  set_solution!(stats, x)
  return stats
end

function get_status(
  nlp::M;
  elapsed_time = 0.0,
  iter = 0,
  optimal = false,
  unbounded = false,
  infeasible = false,
  not_desc = false,
  small_step = false,
  n_iter_since_decrease = 0,
  max_eval = Inf,
  max_time = Inf,
  max_iter = Inf,
  max_decreas_iter = Inf,
) where {M<:AbstractNLPModel}
  if infeasible
    :infeasible
  elseif optimal
    :first_order
  elseif unbounded
    :unbounded
  elseif not_desc
    :not_desc
  elseif small_step
    :small_step
  elseif iter >= max_iter
    :max_iter
  elseif elapsed_time >= max_time
    :max_time
  elseif neval_obj(nlp) >= max_eval && max_eval > -1
    :max_eval
  elseif n_iter_since_decrease ≥ max_decreas_iter
    :infeasible
  else
    :unknown
  end
end
