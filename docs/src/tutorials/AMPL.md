# [AMPL tutorial](@id ampl-tutorial)

This tutorial shows how to solve a model written in [AMPL](https://ampl.com)
with `ExactPenalty.jl`, using
[AmplNLReader.jl](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl).

!!! warning "Inequality Constraints"
    `ExactPenalty.jl` solves problems of the form `minimize f(x) s.t. c(x) = 0`.
    If your AMPL model has inequality constraints, the solver will fail.

## 1. The AMPL model

In this example, we use the Hock--Schittkowski problem **HS6**.

```ampl
var x1 := -1.2;
var x2 := 1;

minimize obj: (1 - x1)^2;

subject to c1: 10 * (x2 - x1^2) = 0;
```

We save this in a `hs6.mod` file.

## 2. Generate the `.nl` file

AMPL compiles a model into a `.nl` file that solvers
read directly, without needing AMPL itself at solve time:

```console
$ ampl -oghs6 assets/hs6.mod
```

This produces file `hs6.nl`.

## 3. Read the model into Julia

```@example ampl
using AmplNLReader

nlp = AmplModel(joinpath(@__DIR__, "assets", "hs6.nl"))

nothing # hide
```

## 4. Solve with ExactPenalty

```@example ampl
using ExactPenalty

stats = L2Penalty(nlp; print_level = 1)

nothing # hide
```

```@example ampl
println("status    : ", stats.status)
println("objective : ", stats.objective)
println("solution  : ", stats.solution)

nothing # hide
```

See the [options reference](../options.md) for the full list of keyword
arguments accepted by the solver.