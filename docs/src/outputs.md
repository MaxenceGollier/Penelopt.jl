# Outputs

Our solver returns a `GenericExecutionStats` object, defined in [SolverCore.jl](https://github.com/JuliaSmoothOptimizers/SolverCore.jl). This section describes the fields of that object populated by `Penelopt.jl`, as well as the console output produced when `print_level > 0`.

```{julia}
julia> stats = L2Penalty(nlp)
```

## The `GenericExecutionStats` Object

* `stats.status::Symbol`: exit status of the algorithm. See [Status](#status) below for the list of possible values.

* `stats.solution::V`: the final iterate $x_k$.

* `stats.objective::T`: the value of $f(x_k)$ at the final iterate.

* `stats.primal_feas::T`: the primal feasibility measure $\lVert c(x_k) \rVert_{\infty}$.

* `stats.dual_feas::T`: the dual feasibility measure $\lVert \nabla f(x_k) + J(x_k)^T y_k \rVert_{\infty}$.

* `stats.multipliers::V`: the vector of Lagrange multiplier estimates $y_k$ at the final iterate.

* `stats.iter::Int`: the number of (*outer*) iterations performed.

* `stats.elapsed_time::Float64`: total elapsed (CPU) time, in seconds.

* `stats.solver_specific::Dict{Symbol,T}`: a dictionary of algorithm-specific quantities, described next.

### `solver_specific` Entries

* `:n_fact::Int`: the cumulative number of matrix factorizations performed by the linear solver over the whole solve.

## Status

`stats.status` can take one of the following values:


| Status | Meaning |
| :------: | :-------- |
| `:first_order` | A first-order stationary point was found within tolerance. |
| `:infeasible` | The problem was declared locally infeasible: see `infeasible_tol` and `infeasible_iter` in the [options](options.md). |
| `:unbounded` | The objective appears to be unbounded below. |
| `:not_desc` | The Moré–Sorensen subsolver could not compute a descent step; see `ms_accept_descent`. |
| `:small_step`| The computed step was numerically insignificant; see `r2n_tiny_step_tol`. |
| `:max_iter` | The iteration limit `max_iter` was reached.|
| `:max_time` | The time limit  `max_time` was reached. |
| `:max_eval` | The objective evaluation limit `max_eval` was reached. |
| `:exception`| An internal exception occurred (e.g., the regularization parameter exceeded `ms_σmax`). |
| `:unknown` | The algorithm has not (yet) satisfied any stopping criterion; this should not appear in a returned `stats` object. |

!!! warning "Infeasibility is a Heuristic"
    A `:infeasible` status means the algorithm *detected* apparent local infeasibility using the heuristic described by `infeasible_tol`; it is not a certificate of infeasibility.

## Console Output

Setting the keyword argument `print_level` to a value $\geq 1$ turns on console logging. Because the solver is organized as nested loops (see [terminology](options.md#Terminology)), `print_level` also controls *how deep* the logging goes:


| `print_level` | Loops logged |
| :--------------: | :------------- |
| `0` | none (default) |
| `1` | outer (penalty) loop |
| `2` | outer + R2N loop |
| `3` | outer + R2N + Moré–Sorensen loop |

The frequency (in iterations) at which each loop prints a line is controlled independently by `verbose`, `r2n_verbose`, and `ms_verbose`.
Additionally, when `print_level ≥ 1`, the solver prints an introduction message at the start of the solve, describing the problem and the solver configuration.

!!! tip "Verifying Your Linear Solver Choice"
    If you set the `linear_solver` [option](options.md#Linear-Solver) and want to confirm it is actually being used, run with `print_level ≥ 1`: the introduction message printed at the start of the solve reports the linear solver library in use.

### Outer-Loop Header

For the outer loop logger, the logger prints the following columns:
* `Iter`: outer iteration counter.
* `sIter`: number of R2N (inner) iterations performed to solve the current penalized subproblem.
* `Objective`: current value of $f(x_k)$.
* `pfeas`: primal feasibility $\lVert c(x_k) \rVert_{\infty}$.
* `dfeas`: dual feasibility $\lVert \nabla f(x_k) + J(x_k)^Ty_k \rVert_{\infty}$.
* `τ`: current penalty parameter.
* `ptol`: primal feasibility tolerance for the current subproblem ($\epsilon^P_k$).
* `dtol`: dual feasibility tolerance for the current subproblem ($\epsilon^D_k$).
* `‖x‖`: norm of the current iterate.

For example,
```@example output-print-level-1
  using CUTEst, Penelopt

  nlp = CUTEstModel("BT7")
  stats = L2Penalty(nlp; print_level = 1)

  finalize(nlp) # hide
```

### Inner (R2N)-Loop Header

For the inner loop logger, the logger prints the following columns:
* `Iter`: R2N iteration counter (within the current penalized subproblem).
* `sIter`: number of Moré–Sorensen iterations used to compute the current step.
* `Objective`: current value of the penalized objective $f(x) + \tau_k \lVert c(x) \rVert_2$.
* `pfeas`, `dfeas`: as above.
* `σ`: current R2N quadratic regularization parameter.
* `ρ`: ratio of actual to first-order predicted decrease for the last step.
* `‖x‖`: norm of the current inner iterate.
* `‖s‖`: norm of the last computed step.

For example,
```@example output-print-level-1
  using CUTEst, Penelopt

  nlp = CUTEstModel("BT7")
  stats = L2Penalty(nlp; print_level = 2)

  finalize(nlp) # hide
```

### MS-Loop Header

!!! warning
    Using `print_level = 3` prints out a lot of information (you are printing 3 nested loops), as you will see below.
    We recommend to use `print_level < 3` first before increasing the value.

For the Moré–Sorensen loop logger, the logger prints the following columns:
* `Iter`: Moré–Sorensen iteration counter (within the current R2N step computation).
* `σ`: current primal regularization parameter of the augmented system.
* `α`: current dual regularization parameter of the augmented system.
* `‖y‖`: norm of the dual step computed by solving the augmented system, compared against the trust-region radius `Δ`.
* `Δ`: current trust-region radius.
* `inertia`: observed inertia $(n_+, n_0, n_-)$ of the augmented system, i.e., the number of positive, zero, and negative eigenvalues.
* `lsolve`: status reported by the linear solver for the last system solved.
* `descent`: whether the computed step was found to be a descent direction for the quadratic model (`true`/`false`); see `ms_accept_descent`.

For example,
```@example output-print-level-1
  using CUTEst, Penelopt

  nlp = CUTEstModel("BT7")
  stats = L2Penalty(nlp; print_level = 3)

  finalize(nlp) # hide
```

!!! tip "Reading `lsolve`"
    The `lsolve` column reports the status returned directly by the chosen `linear_solver` (see [options](options.md#Linear-Solver)) for the corresponding factorization/solve. A value other than `success` typically triggers either a regularization increase or a fallback strategy (e.g., switching MUMPS to an indefinite factorization), and does not necessarily indicate that the overall Moré–Sorensen iteration failed.