using OptimalControl
using NLPModelsIpopt
using Penelopt
using Printf

# ---------------------------------------------------------------------------
# Singular-control benchmark problem
#
# A vehicle moving in the plane with drift, taken from the "Singular control"
# example of OptimalControl.jl:
#
#   https://control-toolbox.org/OptimalControl.jl/stable/example-singular-control.html
#
# IPOPT is reported to struggle on this problem because the optimal
# trajectory contains a singular arc, on which the control cannot be
# recovered directly from the (in)equality multipliers. It is therefore a
# good candidate to stress-test Penelopt.jl on a problem it has not been
# tuned on.
#
# Penelopt currently only supports equality-constrained problems (no bound
# constraints), so the two box constraints `-1 <= u(t) <= 1` and
# `-pi/2 <= theta(t) <= pi/2` -- which, per the discussion with the problem's
# author, are only present to help IPOPT/the direct method converge and are
# not required by the model itself -- are dropped in the version of the
# problem used to compare the two solvers. The original, box-constrained
# problem is kept around and solved with IPOPT alone, to give a baseline of
# how much those bounds actually help.
# ---------------------------------------------------------------------------

"""
    singular_control_ocp(; with_box_constraints::Bool)

Build the singular-control optimal control problem. When
`with_box_constraints` is `true`, the control and state bounds
`-1 <= u(t) <= 1` and `-pi/2 <= theta(t) <= pi/2` from the original example
are included. When it is `false`, they are dropped so that the resulting NLP
is purely equality-constrained, which is required to run Penelopt.
"""
function singular_control_ocp(; with_box_constraints::Bool)
  if with_box_constraints
    ocp = @def begin
      tf ∈ R, variable
      t ∈ [0, tf], time
      q = (x, y, θ) ∈ R³, state
      u ∈ R, control

      -1 ≤ u(t) ≤ 1                     # Control bounds
      -π / 2 ≤ θ(t) ≤ π / 2              # State bounds (helps direct method convergence)

      x(0) == 0
      y(0) == 0
      x(tf) == 1
      y(tf) == 0

      ∂(q)(t) == [cos(θ(t)), sin(θ(t)) + x(t), u(t)]

      tf → min
    end
  else
    ocp = @def begin
      tf ∈ R, variable
      t ∈ [0, tf], time
      q = (x, y, θ) ∈ R³, state
      u ∈ R, control

      x(0) == 0
      y(0) == 0
      x(tf) == 1
      y(tf) == 0

      ∂(q)(t) == [cos(θ(t)), sin(θ(t)) + x(t), u(t)]

      tf → min
    end
  end
  return ocp
end

"""
    discretize_to_nlp(ocp; grid_size = 100, scheme = :trapeze)

Discretize `ocp` following the "NLP manipulations" tutorial
(https://control-toolbox.org/Tutorials.jl/stable/tutorial-nlp.html) and
return the resulting `ADNLPModel`, together with the `docp` and `modeler`
objects needed to turn an NLP solution back into an OCP solution (e.g. to
recover the optimal final time `tf`).
"""
function discretize_to_nlp(ocp; grid_size::Int = 100, scheme::Symbol = :trapeze)
  init = build_initial_guess(ocp, nothing)
  discretizer = OptimalControl.Collocation(grid_size = grid_size, scheme = scheme)
  docp = discretize(ocp, discretizer)
  modeler = OptimalControl.ADNLP(backend = :optimized)
  nlp = nlp_model(docp, init, modeler)
  return nlp, docp, modeler
end

"""
    optimal_time(nlp_stats, docp, modeler)

Recover the optimal final time `tf` from an NLP solution, by rebuilding the
corresponding OCP solution. Returns `NaN` if `nlp_stats` does not describe a
successful solve.
"""
function optimal_time(nlp_stats, docp, modeler)
  try
    ocp_sol = ocp_solution(docp, nlp_stats, modeler)
    return variable(ocp_sol)
  catch
    return NaN
  end
end

function print_row(label, stats)
  status = stats === nothing ? "n/a" : stats.status
  obj = stats === nothing ? NaN : stats.objective
  t = stats === nothing ? NaN : stats.elapsed_time
  it = stats === nothing ? -1 : stats.iter
  @printf("%-28s %-14s %16.8e %10.3f %8d\n", label, status, obj, t, it)
end

function main(; grid_size::Int = 100, tol::Float64 = 1e-6, max_time::Float64 = 300.0)
  ocp_box = singular_control_ocp(with_box_constraints = true)
  ocp_free = singular_control_ocp(with_box_constraints = false)

  nlp_box, docp_box, modeler_box = discretize_to_nlp(ocp_box; grid_size = grid_size)
  nlp_free, docp_free, modeler_free = discretize_to_nlp(ocp_free; grid_size = grid_size)

  @info "Discretized problems" grid_size
  @info "Box-constrained NLP size" nvar = nlp_box.meta.nvar ncon = nlp_box.meta.ncon
  @info "Equality-only NLP size" nvar = nlp_free.meta.nvar ncon = nlp_free.meta.ncon

  # --- IPOPT, original problem with box constraints on u and θ -----------
  @info "Solving the box-constrained problem with IPOPT"
  ipopt_box_stats = ipopt(
    nlp_box,
    print_level = 0,
    tol = tol,
    dual_inf_tol = tol,
    constr_viol_tol = tol,
    compl_inf_tol = tol,
    max_cpu_time = max_time,
    max_iter = typemax(Int32),
  )

  # --- IPOPT, box constraints on u and θ removed --------------------------
  @info "Solving the equality-only problem with IPOPT"
  ipopt_free_stats = ipopt(
    nlp_free,
    print_level = 0,
    tol = tol,
    dual_inf_tol = tol,
    constr_viol_tol = tol,
    compl_inf_tol = tol,
    max_cpu_time = max_time,
    max_iter = typemax(Int32),
  )

  # --- Penelopt, box constraints on u and θ removed -----------------------
  # (L2Penalty only supports equality-constrained problems, hence the
  # box-free formulation.)
  @info "Solving the equality-only problem with Penelopt (L2Penalty)"
  penelopt_free_stats = L2Penalty(
    nlp_free,
    print_level = 0,
    atol = tol,
    rtol = 0.0,
    max_time = max_time,
    max_iter = typemax(Int),
    linear_solver = "mumps",
  )

  println()
  @printf(
    "%-28s %-14s %16s %10s %8s\n",
    "Solver / problem",
    "status",
    "objective",
    "time (s)",
    "iter"
  )
  println("-"^90)
  print_row("IPOPT (box constraints)", ipopt_box_stats)
  print_row("IPOPT (equality only)", ipopt_free_stats)
  print_row("Penelopt (equality only)", penelopt_free_stats)
  println()
  tf_ipopt_box = optimal_time(ipopt_box_stats, docp_box, modeler_box)
  tf_ipopt_free = optimal_time(ipopt_free_stats, docp_free, modeler_free)
  tf_penelopt_free = optimal_time(penelopt_free_stats, docp_free, modeler_free)
  @info "Optimal final time tf" tf_ipopt_box tf_ipopt_free tf_penelopt_free

  return (
    ipopt_box = ipopt_box_stats,
    ipopt_free = ipopt_free_stats,
    penelopt_free = penelopt_free_stats,
  )
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
