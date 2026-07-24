# [CUTEst tutorial](@id cutest-tutorial)

This tutorial shows how to solve a problem from
[CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl)
with `ExactPenalty.jl`.

!!! warning "Inequality Constraints"
    `ExactPenalty.jl` solves problems of the form
    `minimize f(x) s.t. c(x) = 0`.
    Therefore, choose a CUTEst problem containing only equality
    constraints.

## 1. Load a CUTEst problem

In this example, we use the Hock--Schittkowski problem **HS6**.

```@example cutest
using CUTEst

nlp = CUTEstModel("HS6")

nothing # hide
```

## 2. Solve with ExactPenalty

```@example cutest
using ExactPenalty

stats = L2Penalty(nlp; print_level = 1)

nothing # hide
```

```@example cutest
println("status    : ", stats.status)
println("objective : ", stats.objective)
println("solution  : ", stats.solution)

nothing # hide
```

See the [options reference](../options.md) for the full list of keyword
arguments accepted by the solver.

## 3. Finalize the CUTEst model

Once the CUTEst problem has been used, you should finalize it, see the CUTEst [documentation](https://jso.dev/CUTEst.jl/stable/).

```@example cutest
finalize(nlp)

nothing # hide
```