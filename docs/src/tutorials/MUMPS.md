# [MUMPS tutorial](@id mumps-tutorial)

This tutorial shows how to use the [MUMPS](https://mumps-solver.org/index.php?page=doc) linear solver to solve a problem from the
[CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) (see [this](CUTEst.md) tutorial) collection with `Penelopt.jl`.

!!! note "Default Linear Solver"
    `Penelopt.jl` depends directly on [MPI.jl](https://github.com/JuliaParallel/MPI.jl) and [MUMPS.jl](https://github.com/JuliaSmoothOptimizers/MUMPS.jl), and uses MUMPS as its **default** linear solver. You therefore do not need to load any extra package to use it.

## 1. Load a CUTEst problem

In this example, we choose a medium size problem **MSS1**.

```@example mumps
using CUTEst

nlp = CUTEstModel("MSS1")

nothing # hide
```

## 2. Solve with Penelopt

```@example mumps
using Penelopt

stats = L2Penalty(nlp; linear_solver = "mumps", print_level = 1)

nothing # hide
```

```@example mumps
println("status    : ", stats.status)
println("objective : ", stats.objective)
println("solution  : ", stats.solution)

nothing # hide
```

## 3. Finalize the CUTEst model

Once the CUTEst problem has been used, you should finalize it, see the CUTEst [documentation](https://jso.dev/CUTEst.jl/stable/).

```@example mumps
finalize(nlp)

nothing # hide
```

!!! tip "How To Check?"
    You can verify that MUMPS is correctly being used by inspecting the [output](../outputs.md) of the solver with the [option](../options.md) `print_level > 1`.
