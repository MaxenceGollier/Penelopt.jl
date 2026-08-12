using DataFrames
using JLD2
using Measures
using Printf
using SolverBenchmark
using Plots

const METHODS = (:exact, :lbfgs)

function load_stats(dir::AbstractString, stats, suffix = "")

  for method in METHODS

    # Retrieve the number of splits
    file_splits = filter(f -> startswith(f, "stats_$(method)"), readdir(dir))
    n_splits = length(file_splits)

    # Load the first split and initialize the dictionary
    file = joinpath(dir, file_splits[1])
    @info "Loading $file"
    dict = load(file)["stats"]
    println(dict)

    # Load the remaining splits and concatenate the data
    for split in 2:n_splits
      file = joinpath(dir, file_splits[split])
      @info "Loading $file"
      dict_split = load(file)["stats"]
      for key in keys(dict)
        append!(dict[key], dict_split[key])
      end
    end

    for key in keys(dict)
      new_key = Symbol("$(key)$suffix")
      stats[new_key] = dict[key]
    end
  end

  return stats
end

function pairwise_plot(stats, keys; compare_n_fact = false)
  solved(df) = (df.status .== :first_order) # TODO: add infeasible problems
  costs = [
    df -> .!solved(df) * Inf + df.elapsed_time,
    df -> .!solved(df) * Inf + df.neval_obj,
    df -> .!solved(df) * Inf + df.neval_grad,
  ]
  costnames = ["CPU Time", "# Objective Evals", "# Gradient Evals"]

  if compare_n_fact &&
     hasproperty(stats[keys[1]], :n_fact) &&
     hasproperty(stats[keys[2]], :n_fact)
    costnames[1] = "# Factorizations"
    costs[1] = df -> .!solved(df) * Inf .+ coalesce.(df.n_fact, Inf)
  end

  stats_subset = filter(kv -> kv[1] in keys, stats)

  # split the stat names. the first part is always l2penalty, then the method used, then the tolerance and finally the branch.
  parts_1 = Symbol.(split(string(keys[1]), "_"))
  parts_2 = Symbol.(split(string(keys[2]), "_"))

  models = Dict(:exact => "∇²L(x, y)", :lbfgs => "BFGS")
  precision = Dict(:imprecise => "1e-3", :precise => "1e-9")

  suptitle = "\nHessian model: Bₖ(x) = " * models[parts_1[2]]
  p = profile_solvers(
    stats_subset,
    costs,
    costnames;
    suptitle = suptitle,
    xlabel = "",
    ylabel = "",
  )
  p.subplots[2][:legend_position] = :bottomright
  p.subplots[3][:legend_position] = :bottomright

  compare_with = (parts_1[end] == :reference || parts_2[end] == :reference) ? :main : :ipopt
  p.series_list[1][:label] =
    Symbol.(split(string(p.series_list[1][:label]), "_"))[end] == :current ? :current :
    compare_with
  p.series_list[2][:label] =
    Symbol.(split(string(p.series_list[2][:label]), "_"))[end] == :current ? :current :
    compare_with
  p.series_list[3][:label] =
    Symbol.(split(string(p.series_list[3][:label]), "_"))[end] == :current ? :current :
    compare_with
  p.series_list[4][:label] =
    Symbol.(split(string(p.series_list[4][:label]), "_"))[end] == :current ? :current :
    compare_with
  p.series_list[5][:label] =
    Symbol.(split(string(p.series_list[5][:label]), "_"))[end] == :current ? :current :
    compare_with
  p.series_list[6][:label] =
    Symbol.(split(string(p.series_list[6][:label]), "_"))[end] == :current ? :current :
    compare_with

  p.series_list[1][:linecolor] =
    Symbol.(split(string(p.series_list[1][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[2][:linecolor] =
    Symbol.(split(string(p.series_list[2][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[3][:linecolor] =
    Symbol.(split(string(p.series_list[3][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[4][:linecolor] =
    Symbol.(split(string(p.series_list[4][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[5][:linecolor] =
    Symbol.(split(string(p.series_list[5][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[6][:linecolor] =
    Symbol.(split(string(p.series_list[6][:label]), "_"))[end] == :current ? :blue : :red

  return p
end

current_dir = joinpath("artifacts", "current")
reference_dir = joinpath("artifacts", "reference")

stats = Dict{Symbol,DataFrame}()

@info "Loading current benchmark results"
load_stats(current_dir, stats, "_current")

@info "Loading reference benchmark results"
load_stats(reference_dir, stats, "_reference")

p = plot(
  pairwise_plot(
    stats,
    [:l2penalty_exact_reference, :l2penalty_exact_current],
    compare_n_fact = true,
  ),
  pairwise_plot(
    stats,
    [:l2penalty_lbfgs_reference, :l2penalty_lbfgs_current],
    compare_n_fact = true,
  ),
  layout = (2, 1),
  size = (1920, 1080),
)

mkpath("benchmark/result")
savefig(p, "benchmark/result/benchmark_comparison.svg")

# Plot IPOPT
ipopt_dir = joinpath("artifacts", "ipopt")

@info "Loading ipopt benchmark results"
ipopt_stats = load(joinpath(ipopt_dir, "stats_ipopt.jld2"))["stats"]
for key in keys(ipopt_stats)
  stats[key] = ipopt_stats[key]
end

p = plot(
  pairwise_plot(stats, [:l2penalty_exact_current, :ipopt_exact]),
  pairwise_plot(stats, [:l2penalty_lbfgs_current, :ipopt_lbfgs]),
  layout = (2, 1),
  size = (1920, 1080),
)

savefig(p, "benchmark/result/benchmark_comparison_ipopt.svg")

@info "Infeasibility results\n"

function infeasibility_pair(stats, keys)
  df_1 = stats[keys[1]]
  df_2 = stats[keys[2]]

  parts_1 = Symbol.(split(string(keys[1]), "_"))
  parts_2 = Symbol.(split(string(keys[2]), "_"))

  @assert parts_2[1] == :ipopt

  @info "Checking infeasibility results for $(parts_1[2]) Hessian approximation."
  for i = 1:nrow(df_1)
    @assert df_1[i, :name] == df_2[i, :name]
    if df_1[i, :status] == :infeasible && df_2[i, :status] == :infeasible
      @info "IPOPT and L2Penalty both declared $(df_1[i, :name]) infeasible"
    elseif df_1[i, :status] == :infeasible
      @info "L2Penalty declared $(df_1[i, :name]) infeasible, but IPOPT terminated with status $(df_2[i, :status])"
    elseif df_2[i, :status] == :infeasible
      @info "IPOPT declared $(df_1[i, :name]) infeasible, but L2Penalty terminated with status $(df_1[i, :status])"
    end
  end
  @info ""
end

infeasibility_pair(stats, [:l2penalty_exact_current, :ipopt_exact])
