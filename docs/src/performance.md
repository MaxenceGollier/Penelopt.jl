# Performance

## Preallocation

When calling our solver, `ExactPenalty.jl` starts with an *allocation* phase where all of the vectors, matrices, and sub-solvers needed to run the algorithm are built.
The workspace is allocated in a [struct](https://docs.julialang.org/en/v1/manual/types/#Composite-Types) called `L2PenaltySolver`.
Our solver then proceeds to solve the provided `nlp`, in an allocation-free function named `solve!`.

In particular, this means that once a solver has been constructed, it can be reused across several calls to `solve!` without allocating any new memory, provided the problem dimensions ($n$ variables, $m$ constraints) stay the same.
This is particularly useful:

* when solving the same problem repeatedly from different starting points (e.g., in a multistart strategy);
* when solving a sequence of closely related problems (e.g., inside a loop, for bilevel or parametric optimization);
* when you want to avoid the garbage collector getting in the way of a tight benchmarking loop.

The convenience call
```{julia}
julia> stats = L2Penalty(nlp)
```
internally does exactly the following:
```{julia}
julia> solver = L2PenaltySolver(nlp)
julia> stats = ExactPenaltyExecutionStats(nlp)
julia> solve!(solver, nlp, stats)
```
Separating these steps is the key to avoiding allocations: `L2PenaltySolver(nlp)` performs *all* the allocations up front (workspaces, factorization buffers, sub-solver states, etc.), and every subsequent call to `solve!` reuses that memory in place.

```{julia}
julia> using ExactPenalty

# Allocate the solver and the stats object once.
julia> solver = L2PenaltySolver(nlp)
julia> stats = ExactPenaltyExecutionStats(nlp)

# Solve as many times as needed;
julia> solve!(solver, nlp, stats)
julia> solve!(solver, nlp, stats; x = x1)
julia> solve!(solver, nlp, stats; x = x2)
```

!!! warning "Resetting internal state"
    Some internal quantities of the solver (such as quasi-Newton
    approximations, the watchdog checkpoint, and factorization counters)
    persist between calls to `solve!`. Therefore, we strongly recommend calling [`SolverCore.reset!`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl)
    between calls to `solve!`.
    ```{julia}
    julia> solve!(solver, nlp, stats)
    julia> reset!(solver)
    julia> solve!(solver, nlp, stats; x = x1)
    julia> ...
    ```

!!! warning "Keyword arguments"
    Some [options](options.md) require allocations and are therefore passed to the solver structure construction call.
    The following options **need** to be passed to the solver constructor (passing them to `solve!` will cause failure).
    * `r2n_m_monotone::Int = 12`;
    * `linear_solver::Sring = "ldlt"`.
    For clarity, if you want to modify these options, while preallocating the workspace, you should do
    ```{julia}
    julia> solver = L2PenaltySolver(nlp, r2n_m_monotone = 6, linear_solver = "mumps")
    julia> ...
    ```

### The `L2PenaltySolver` structure

The fields for the `L2PenaltySolver` are:

| Field | Type | Description |
|:------|:-----|:-------------|
| `x` | `V` | current outer iterate $x_k$ |
| `xn` | `V` | (*internal*) trial/next iterate buffer |
| `y` | `V` | current Lagrange multiplier estimate $y_k$ |
| `cn` | `V` | (*internal*) buffer for the constraint values at a trial point |
| `dual_res` | `V` | buffer for the dual residual $\nabla f(x_k) + J(x_k)^Ty_k$ |
| `s` | `V` | (*internal*) buffer for the computed step |
| `s0` | `V` | (*internal*) buffer used when computing least-squares multipliers |
| `∇fk` | `V` | buffer for $\nabla f(x_k)$ |
| `temp_b` | `V` | (*internal*) scratch buffer of length $m$, used e.g. by the least-squares multiplier computation |
| `subsolver` | `PenaltyR2NSolver` | the R2N (inner-loop) solver structure used to (approximately) minimize the current penalized subproblem $f(x) + \tau_k\lVert c(x) \rVert_2$ |
| `subpb` | `L2PenalizedProblem` | the penalized subproblem $f(x) + \tau_k\lVert c(x)\rVert_2$ associated with `nlp` |
| `substats` | `GenericExecutionStats` | the statistics object updated by `subsolver` at every R2N iteration |

We mark some fields as *internal* which means that we discourage users to read/modify these in [callbacks](callbacks.md) as they are only used for internal computations.

Here, `T` is the scalar type and `V` is the vector type of the problem (by
default `T == Float64` and `V == Vector{Float64}`; see the [note on
parametric types](options.md) on the options page).

### Nested solver structures

Because the algorithm is organized in nested loops (see the [options
terminology](options.md#Terminology)), `solver.subsolver` and
`solver.subsolver.subsolver` expose the state of the inner loops.

* `solver.subsolver::PenaltyR2NSolver` holds the state needed to minimize a
  *single* penalized subproblem $f(x) + \tau\lVert c(x)\rVert_2$ for a fixed
  $\tau$: the current inner iterate `xk`, its multiplier estimate `y`, the
  non-monotone objective history `m_fh_hist`, the watchdog `checkpoint`,
  and its own `subsolver::MoreSorensenSolver` and `subpb::ShiftedL2PenalizedProblem`.
* `solver.subsolver.subsolver::MoreSorensenSolver` holds the buffers needed
  to compute a single step by (approximately) solving the trust-region
  subproblem via the Moré–Sorensen method: the right-hand side/solution
  buffers `u1`, `u2`, `x1`, `x2`, the assembled KKT-like matrix `H`, and the
  `workspace` used by the chosen [linear solver](options.md#Linear-Solver)
  (e.g., `PenaltyLDLTWorkspace`, `PenaltyMUMPSWorkspace`, ...).

All three levels can be constructed once (`L2PenaltySolver`,
`PenaltyR2NSolver`, or `MoreSorensenSolver`, respectively) and reused across
solves, following the same two-call pattern described above.

## Quasi-Newton Approximations

If the Hessian of the Lagrangian of your nonlinear programming problem is dense, ill-conditionned, expensive to compute, or inaccessible, you may be interested in replacing it with a quasi-Newton approximation.

`ExactPenalty.jl` offers you the possibility to run the optimization process with a [Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) approximation.
You simply need to call the function `CompactBFGSModel` on your `nlp`!

```@example bfgs
using CUTEst, ExactPenalty

# Construct your NLP model
nlp = CUTEstModel("HS6")

# Construct your BFGS model
bfgs_nlp = CompactBFGSModel(nlp)

stats = L2Penalty(bfgs_nlp; print_level = 1)

finalize(nlp) # hide
```

!!! warning "Default options"
    Some default [options](options.md) have a different value when using our solver with a quasi-Newton approximation.
    Those options are
    * `r2n_η2::T = 0.1` which default to `0.9` with a quasi-Newton approximation;
    * `ms_αmin1::T = eps(T)^(0.6)` which default to `eps(T)^(0.8)` with a quasi-Newton approximation.

!!! tip "L-BFGS options"
    The `CompactBFGSModel` constructor supports keyword arguments to customize your approximation.
    Those are
    * `mem::Int = 5`: memory parameter for the limited-memory approximation.
    * `scaling::Bool = true`: whether we scale $B_0 = \gamma I$ with $\gamma = y^Ty / s^T y$, where $s$ is the step and $y$ is the difference of the two last gradients of the Lagrangian.
    * `max_skip::Int = 2` (*advanced*): if we skipped a pair $(s, y)$ more than `max_skip` times in a row, we reset the approximation. 