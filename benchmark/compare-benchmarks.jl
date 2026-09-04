using DataFrames
using JLD2
using Measures
using Printf
using SolverBenchmark
using Plots

include(joinpath(@__DIR__, "benchmark-utils.jl"))
include(joinpath(@__DIR__, "infeasibility-checker.jl"))

# Certifying re-solves each flagged problem from scratch - expensive (a
# recent run took ~1h30). Off by default; CI sets CERTIFY_INFEASIBILITY=true
# via the "certify infeasibility" label. `current` is certified fresh here;
# `reference`/`ipopt` are precomputed baselines loaded from
# certify-infeasibility.jl's output instead of re-solved every PR run.
const CERTIFY_INFEASIBILITY = lowercase(get(ENV, "CERTIFY_INFEASIBILITY", "false")) == "true"

function pairwise_plot(
  stats,
  keys;
  compare_n_fact = false,
  certified_infeasible = Dict{Symbol,Set{String}}(),
)
  df_1 = stats[keys[1]]
  df_2 = stats[keys[2]]

  # pick the right set by matching df to df_1/df_2 by identity
  cert_1 = get(certified_infeasible, keys[1], Set{String}())
  cert_2 = get(certified_infeasible, keys[2], Set{String}())
  solved(df) =
    (df.status .== :first_order) .|
    ((df.status .== :infeasible) .& in.(df.name, Ref(df === df_1 ? cert_1 : cert_2)))
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

"""
    load_precomputed_certification(dir)

Find `infeasibility_certification.jld2` (produced by
certify-infeasibility.jl) anywhere under `dir` and return the saved
DataFrame, or `nothing` if none is found.
"""
function load_precomputed_certification(dir::AbstractString)
  cert_file = nothing
  for (root, _, files) in walkdir(dir)
    "infeasibility_certification.jld2" in files &&
      (cert_file = joinpath(root, "infeasibility_certification.jld2"))
  end

  if cert_file === nothing
    @warn "No precomputed infeasibility certification found under $(dir)."
    return nothing
  end

  @info "Loading precomputed infeasibility certification from $cert_file"
  return load(cert_file)["infeasibility_certification"]
end

"""
    lookup_certification(report, name, hessian)

Look up `name`'s certification (for `hessian`) in a precomputed `report`.
Returns "N/A" if `report` is `nothing` or doesn't cover `name`.
"""
function lookup_certification(report, name::AbstractString, hessian::Symbol)
  report === nothing && return "N/A"
  idx = findfirst(row -> row.name == name && row.hessian == hessian, eachrow(report))
  idx === nothing && return "N/A"
  return report[idx, :certified_locally_infeasible]
end

"""
    register_precomputed_certified!(certified_infeasible, report, key)

Add every problem in `report` certified locally infeasible to
`certified_infeasible[key]`. Used for `reference`, which doesn't go
through `infeasibility_pair`.
"""
function register_precomputed_certified!(
  certified_infeasible::Dict{Symbol,Set{String}},
  report,
  key::Symbol,
)
  report === nothing && return certified_infeasible
  hessian = Symbol.(split(string(key), "_"))[2]
  for row in eachrow(report)
    if row.hessian == hessian && row.certified_locally_infeasible === true
      push!(get!(() -> Set{String}(), certified_infeasible, key), row.name)
    end
  end
  return certified_infeasible
end

function infeasibility_pair(
  stats,
  keys,
  certified_infeasible::Dict{Symbol,Set{String}},
  ipopt_certification,
)
  df_1 = stats[keys[1]]
  df_2 = stats[keys[2]]

  parts_1 = Symbol.(split(string(keys[1]), "_"))
  parts_2 = Symbol.(split(string(keys[2]), "_"))

  @assert parts_2[1] == :ipopt

  hessian = parts_1[2]

  @info "Checking infeasibility results for $(hessian) Hessian approximation."
  if !CERTIFY_INFEASIBILITY
    @info "CERTIFY_INFEASIBILITY is off: listing flagged problems without certifying them."
  end

  rows = NamedTuple[]
  for i = 1:nrow(df_1)
    @assert df_1[i, :name] == df_2[i, :name]
    name = df_1[i, :name]
    l2penalty_status = df_1[i, :status]
    ipopt_status = df_2[i, :status]

    l2penalty_status != :infeasible && ipopt_status != :infeasible && continue

    # `current` certified fresh; `ipopt` looked up from precomputed data.
    # "N/A" = not attempted/available; `missing` = attempted, inconclusive.
    l2penalty_certified =
      (l2penalty_status == :infeasible && CERTIFY_INFEASIBILITY) ?
      certify_local_infeasibility(name, keys[1]) : "N/A"
    ipopt_certified =
      (ipopt_status == :infeasible && CERTIFY_INFEASIBILITY) ?
      lookup_certification(ipopt_certification, name, hessian) : "N/A"

    if l2penalty_certified === true
      push!(get!(() -> Set{String}(), certified_infeasible, keys[1]), name)
    end
    if ipopt_certified === true
      push!(get!(() -> Set{String}(), certified_infeasible, keys[2]), name)
    end

    push!(
      rows,
      (
        name = name,
        hessian = hessian,
        l2penalty_status = l2penalty_status,
        ipopt_status = ipopt_status,
        l2penalty_certified_locally_infeasible = l2penalty_certified,
        ipopt_certified_locally_infeasible = ipopt_certified,
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

# Must run before the performance profiles: certified problems affect how
# :infeasible is treated in the profiles' costs.
@info "Infeasibility results\n"

# reference/ipopt: precomputed baselines, loaded not re-solved. Skipped
# when CERTIFY_INFEASIBILITY is off.
reference_certification =
  CERTIFY_INFEASIBILITY ? load_precomputed_certification(reference_dir) : nothing
ipopt_certification = CERTIFY_INFEASIBILITY ? load_precomputed_certification(ipopt_dir) : nothing

# certified_infeasible[key]: problems certified locally infeasible for that
# specific run - never credited across solvers, since they can land on
# different points.
certified_infeasible = Dict{Symbol,Set{String}}()

# reference isn't paired against ipopt below, so register it directly.
register_precomputed_certified!(certified_infeasible, reference_certification, :l2penalty_exact_reference)
register_precomputed_certified!(certified_infeasible, reference_certification, :l2penalty_lbfgs_reference)

exact_report = infeasibility_pair(
  stats,
  [:l2penalty_exact_current, :ipopt_exact],
  certified_infeasible,
  ipopt_certification,
)
lbfgs_report = infeasibility_pair(
  stats,
  [:l2penalty_lbfgs_current, :ipopt_lbfgs],
  certified_infeasible,
  ipopt_certification,
)
infeasibility_report = vcat(exact_report, lbfgs_report)

@info "Infeasibility certification results:\n" *
      sprint(io -> show(io, infeasibility_report; allrows = true, allcols = true))

mkpath("benchmark/result")
@save "benchmark/result/infeasibility_report.jld2" infeasibility_report

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
