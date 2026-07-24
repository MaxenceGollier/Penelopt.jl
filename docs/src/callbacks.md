# Callbacks
 
`ExactPenalty.jl` lets you hook into the solve process at each iteration through a
*callback* function. This is useful to log custom information, record
history, plot progress in real time, or implement your own stopping
criterion.
  
The signature of the *callback* function is
```julia
callback(nlp, solver, stats)
```
and its return value is ignored. 
The call to the callback function is the last step of each iteration.
Therefore, you can customize the behavior of the algorithm by implementing your customization into the callback.
Below are a few typical examples of callback customizations.
 
### Example 1 : Plot the Objective Value History
```@example cb-1
using CUTEst, ExactPenalty, Plots
 
# Define the nonlinear program
nlp = CUTEstModel("HS26")

# Define the callback: you can use variables defined in the current scope
objvals = Float64[]

function my_callback(nlp, solver, stats)
  push!(objvals, stats.objective)
end
 
# Solve the problem
stats = L2Penalty(nlp, callback = my_callback)
finalize(nlp) # hide

# Plot the objective values
plot_kwargs = ( # hide
    xlabel = "k", # hide
    ylabel = "f(xₖ)", # hide
    title = "Objective Value History", # hide
    yscale = :log10, # hide
    lw = 2.5, # hide
    legend = false, # hide
    framestyle = :box, # hide
    gridalpha = 0.25, # hide
    xticks = -1:2:length(objvals), # hide
    yticks = 10.0 .^ (-15:3:1), # hide
    tickfontsize = 10, # hide
    guidefontsize = 12, # hide
    titlefontsize = 14, # hide
    size = (650, 400), # hide
) # hide
plot(objvals; plot_kwargs...)
```
 
!!! warning
    Note that the callback acts at the end of each *outer* loop iteration.
    Refer to the [terminology](options.md#Terminology) section for details on what this means.

### Example 2 : Stop the Algorithm Early

Setting
```julia
stats.status = :user
```
inside the callback will cause the algorithm to stop immediately after the
callback returns. Use this to implement custom stopping criteria (e.g., a
target objective value, a wall-clock budget managed externally, or an
interactive "stop" signal).
 
```@example cb-2
using CUTEst, ExactPenalty

# Define the nonlinear program
nlp = CUTEstModel("HS26")

# Define the callback
# We stop when the objective function is below 1e-3.
function stop_early(nlp, solver, stats)
  if stats.objective < 1e-3
    println("returning on user request...")
    stats.status = :user
  end
end

# Solve the problem
stats = L2Penalty(nlp, callback = stop_early, print_level = 1)

finalize(nlp) # hide
```
 
## What Can You Access
 
All the information relevant to the current state of the algorithm is
available through `nlp`, `solver`, and `stats`. 

In particular:

* `nlp`: the [`AbstractNLPModel`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) object that contains information relative to the nonlinear program solved by `ExactPenalty.jl`. You can for example access the [problem meta](https://jso.dev/NLPModels.jl/stable/reference/#NLPModels.NLPModelMeta) or the [problem counters](https://jso.dev/NLPModels.jl/stable/tools/#Functions-evaluations) in the callback.
* `solver`: the `ExactPenaltySolver` structure containing all allocated objects used during the optimization process. Refer to the [performance](performance.md) section of the documentation for a list of information that you can access through this structure.
* `stats`: the [`GenericExecutionStats`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl)
  object that will eventually be returned by `L2Penalty`. You can refer to [this section](outputs.md#the-genericexecutionstats-object) for a list of the information contained in this object.