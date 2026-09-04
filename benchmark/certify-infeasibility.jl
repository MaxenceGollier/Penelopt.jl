using DataFrames
using JLD2

include(joinpath(@__DIR__, "benchmark-utils.jl"))
include(joinpath(@__DIR__, "infeasibility-checker.jl"))

# Precomputes the infeasibility certification of a *fixed* baseline (the
# reference L2Penalty run saved by SaveBenchmark.yml, or the IPOPT run
# saved by RunIpoptBenchmark.yml), so that PR comparisons in
# compare-benchmarks.jl can load these results (via
# load_precomputed_certification) instead of re-certifying two baselines
# that don't change from one PR to the next on every single PR run.
#
# Expects the stats_*.jld2 files this run itself just produced to already
# be sitting in benchmark/result (i.e. run this *after* the benchmark step,
# before uploading the artifact - the CI jobs merge every matrix split's
# stats first via download-artifact so all of a Hessian model's problems
# are certified together).

stats = Dict{Symbol,DataFrame}()
load_stats("benchmark/result", stats)

reports = DataFrame[]
for key in keys(stats)
  push!(reports, certify_own_infeasibility(stats, key))
end

infeasibility_certification =
  isempty(reports) ?
  DataFrame(
    name = String[],
    hessian = Symbol[],
    status = Symbol[],
    certified_locally_infeasible = Union{Bool,Missing}[],
  ) : vcat(reports...)

@info "Infeasibility certification results:\n" *
      sprint(io -> show(io, infeasibility_certification; allrows = true, allcols = true))

mkpath("benchmark/result")
@save "benchmark/result/infeasibility_certification.jld2" infeasibility_certification
