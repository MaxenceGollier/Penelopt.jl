# [JuMP tutorial](@id jump-tutorial)

This tutorial shows how to solve a model written in [JuMP](https://jump.dev/JuMP.jl/stable/) with `ExactPenalty.jl`, 
using [NLPModelsJuMP.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl).

!!! warning "Inequality Constraints"
    `ExactPenalty.jl` solves problems of the form `minimize f(x) s.t. c(x) = 0`.
    If your JuMP model has inequality constraints, the solver will fail.

## 1. Build the JuMP model

In this example, we use the Hock--Schittkowski problem **HS6**.

```@example jump
using JuMP
model = Model()

@variable(model, x1, start = -1.2)
@variable(model, x2, start = 1.0)

@objective(model, Min, (1 - x1)^2)

@constraint(model, 10 * (x2 - x1^2) == 0)

nothing # hide
```

## 2. Wrap it as an `NLPModel`

```@example jump
using NLPModelsJuMP

nlp = MathOptNLPModel(model)

nothing # hide
```

## 3. Solve with ExactPenalty

```@example jump
using ExactPenalty

stats = L2Penalty(nlp; print_level = 1)

nothing # hide
```

```@example jump
println("status    : ", stats.status)
println("objective : ", stats.objective)
println("solution  : ", stats.solution)

nothing # hide
```

See the [options reference](../options.md) for the full list of keyword
arguments accepted by the solver.