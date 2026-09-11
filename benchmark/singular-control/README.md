# Singular-control benchmark

This benchmark compares `Penelopt.jl` against `IPOPT` on the **singular
control** example from
[`OptimalControl.jl`](https://control-toolbox.org/OptimalControl.jl/stable/example-singular-control.html):
a time-optimal transfer of a vehicle moving in the plane with drift, whose
optimal trajectory contains a singular arc. IPOPT is known to have some
difficulty with this kind of arc, which makes it a useful problem to test
Penelopt on, outside of the CUTEst problems it has already been benchmarked
against.

The optimal control problem is discretized into a nonlinear program using
`OptimalControl.jl`'s low-level API, following the
["NLP manipulations" tutorial](https://control-toolbox.org/Tutorials.jl/stable/tutorial-nlp.html):
`build_initial_guess` → `discretize` (trapezoidal collocation) →
`nlp_model` (`ADNLP` modeler), which produces an `ADNLPModel` that can be
handed directly to any `NLPModels`-compatible solver.

Since `Penelopt.L2Penalty` currently only supports equality-constrained
problems, the two box constraints of the original example,
`-1 ≤ u(t) ≤ 1` and `-π/2 ≤ θ(t) ≤ π/2`, are dropped before handing the
problem to Penelopt. These bounds only help the *direct* method converge to
the right branch of the solution and are not otherwise required by the
model, so this does not change the underlying problem being solved. Both
versions of the problem (with and without the box constraints) are solved
with IPOPT, and the box-free version is additionally solved with Penelopt,
so the three runs can be compared side by side.

## Running

From the root of the repository:

```bash
julia --project=benchmark/singular-control -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/singular-control benchmark/singular-control/singular-control.jl
```

This prints, for each solver/problem combination, the exit status, final
objective value (`tf`), solve time, and iteration count, followed by the
recovered optimal final time `tf` for each successful solve.

## Files

- `singular-control.jl`: builds the two OCP formulations, discretizes them,
  and runs IPOPT and Penelopt.
- `Project.toml`: a self-contained environment for this benchmark; it uses
  the `Penelopt` package from the repository root via `[sources]` so it
  always tracks the current state of the code under review.
