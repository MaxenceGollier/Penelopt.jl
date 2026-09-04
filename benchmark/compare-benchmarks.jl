using DataFrames
using JLD2
using Measures
using Printf
using SolverBenchmark
using Plots

include(joinpath(@__DIR__, "benchmark-utils.jl"))
include(joinpath(@__DIR__, "infeasibility-checker.jl"))
include(joinpath(@__DIR__, "compare-utils.jl"))

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
