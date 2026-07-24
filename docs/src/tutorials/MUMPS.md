# [MUMPS tutorial](@id mumps-tutorial)

This tutorial shows how to use the [MUMPS](https://mumps-solver.org/index.php?page=doc) linear solver to solve a problem from the
[CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) (see [this](CUTEst.md) tutorial) collection with `ExactPenalty.jl`.

!!! warning "Extensions"
    `ExactPenalty.jl` uses an [extension](https://docs.julialang.org/en/v1/manual/code-loading/#man-extensions) to load the [MUMPS](https://mumps-solver.org/index.php?page=doc) linear solver. Therefore, you **need** to load both [MPI.jl](https://github.com/JuliaParallel/MPI.jl) and [MUMPS.jl](https://github.com/JuliaSmoothOptimizers/MUMPS.jl). Our algorithm will throw a warning and switch to the default [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) package if you try to use MUMPS without loading the required packages.

## 1. Load a CUTEst problem

In this example, we choose a medium size problem **MSS1**.

```@example mumps
using CUTEst

nlp = CUTEstModel("MSS1")

nothing # hide
```

## 2. Load the MUMPS packages

```@example mumps
using MPI, MUMPS
```

## 3. Solve with ExactPenalty

```@example mumps
using ExactPenalty

stats = L2Penalty(nlp; linear_solver = "mumps", print_level = 1)

nothing # hide
```

```@example mumps
println("status    : ", stats.status)
println("objective : ", stats.objective)
println("solution  : ", stats.solution)

nothing # hide
```

## 4. Finalize the CUTEst model

Once the CUTEst problem has been used, you should finalize it, see the CUTEst [documentation](https://jso.dev/CUTEst.jl/stable/).

```@example mumps
finalize(nlp)

nothing # hide
```

!!! tip "How To Check?"
    You can verify that MUMPS is correctly being used by inspecting the [output](../outputs.md) of the solver with the [option](../options.md) `print_level > 1`.