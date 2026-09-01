using DataFrames
using JLD2
using Measures
using Printf
using SolverBenchmark
using Plots

include(joinpath(@__DIR__, "infeasibility-checker.jl"))

const METHODS = (:exact, :lbfgs)

function load_stats(dir::AbstractString, stats, suffix = "")

  for method in METHODS

    @info "Loading $(method) benchmark results"

    file_splits = String[]

    for (root, _, files) in walkdir(dir)
      for file in files
        if (
          startswith(file, "stats_$(method)") ||
          (startswith(file, "stats_ipopt_$(method)") && suffix == "")
        ) && occursin(r"\d+\.jld2$", file)
          push!(file_splits, joinpath(root, file))
        end
      end
    end

    sort!(file_splits)

    n_splits = length(file_splits)

    # Load the first split and initialize the dictionary
    file = file_splits[1]
    @info "Loading $file"
    dict = load(file)["stats"]

    # Load the remaining splits and concatenate the data
    for split = 2:n_splits
      file = file_splits[split]
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

function pairwise_plot(
  stats,
  keys;
  compare_n_fact = false,
  certified_infeasible = Set{String}(),
)
  solved(df) =
    (df.status .== :first_order) .|
    ((df.status .== :infeasible) .& in.(df.name, Ref(certified_infeasible)))
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

function infeasibility_pair(stats, keys)
  df_1 = stats[keys[1]]
  df_2 = stats[keys[2]]

  parts_1 = Symbol.(split(string(keys[1]), "_"))
  parts_2 = Symbol.(split(string(keys[2]), "_"))

  @assert parts_2[1] == :ipopt

  hessian = parts_1[2]

  @info "Checking infeasibility results for $(hessian) Hessian approximation."

  rows = NamedTuple[]
  for i = 1:nrow(df_1)
    @assert df_1[i, :name] == df_2[i, :name]
    name = df_1[i, :name]
    l2penalty_status = df_1[i, :status]
    ipopt_status = df_2[i, :status]

    l2penalty_status != :infeasible && ipopt_status != :infeasible && continue

    # Certify against whichever run actually declared infeasibility. If both
    # did, we only need to reproduce one of them (L2Penalty's, since that is
    # the run whose status feeds the performance profiles).
    key = l2penalty_status == :infeasible ? keys[1] : keys[2]
    certified = certify_local_infeasibility(name, key)

    push!(
      rows,
      (
        name = name,
        hessian = hessian,
        l2penalty_status = l2penalty_status,
        ipopt_status = ipopt_status,
        certified_locally_infeasible = certified,
      ),
    )
  end

  return DataFrame(rows)
end

current_dir = joinpath("artifacts", "current")
reference_dir = joinpath("artifacts", "reference")
ipopt_dir = joinpath("artifacts", "ipopt")

stats = Dict{Symbol,DataFrame}()

@info "Loading current benchmark results"
load_stats(current_dir, stats, "_current")

@info "Loading reference benchmark results"
load_stats(reference_dir, stats, "_reference")

@info "Loading ipopt benchmark results"
load_stats(ipopt_dir, stats, "")

# Infeasibility check must run before the performance profiles, since its
# result (which problems are certified locally infeasible) feeds into how
# the profiles' costs treat the :infeasible status.
@info "Infeasibility results\n"

exact_report = infeasibility_pair(stats, [:l2penalty_exact_current, :ipopt_exact])
lbfgs_report = infeasibility_pair(stats, [:l2penalty_lbfgs_current, :ipopt_lbfgs])
infeasibility_report = vcat(exact_report, lbfgs_report)

@info "Infeasibility certification results:\n" *
      sprint(show, infeasibility_report; allrows = true, allcols = true)

mkpath("benchmark/result")
@save "benchmark/result/infeasibility_report.jld2" infeasibility_report

# Only L2Penalty's own :infeasible calls affect the performance profiles, and
# only when the certificator conclusively (i.e. not `missing`) confirmed the
# point is locally infeasible.
certified_infeasible = Set(
  row.name for row in eachrow(infeasibility_report) if
  row.l2penalty_status == :infeasible && row.certified_locally_infeasible === true
)

p = plot(
  pairwise_plot(
    stats,
    [:l2penalty_exact_reference, :l2penalty_exact_current],
    compare_n_fact = true,
    certified_infeasible = certified_infeasible,
  ),
  pairwise_plot(
    stats,
    [:l2penalty_lbfgs_reference, :l2penalty_lbfgs_current],
    compare_n_fact = true,
    certified_infeasible = certified_infeasible,
  ),
  layout = (2, 1),
  size = (1920, 1080),
)

savefig(p, "benchmark/result/benchmark_comparison.svg")

# Plot IPOPT
p = plot(
  pairwise_plot(
    stats,
    [:l2penalty_exact_current, :ipopt_exact],
    certified_infeasible = certified_infeasible,
  ),
  pairwise_plot(
    stats,
    [:l2penalty_lbfgs_current, :ipopt_lbfgs],
    certified_infeasible = certified_infeasible,
  ),
  layout = (2, 1),
  size = (1920, 1080),
)

savefig(p, "benchmark/result/benchmark_comparison_ipopt.svg")
