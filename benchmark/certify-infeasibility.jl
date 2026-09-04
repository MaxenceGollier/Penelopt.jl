using DataFrames
using JLD2

include(joinpath(@__DIR__, "utils", "benchmark-utils.jl"))
include(joinpath(@__DIR__, "utils", "infeasibility-checker.jl"))

# Precomputes a fixed baseline's (reference or ipopt) infeasibility
# certification, saved once and loaded by compare-benchmarks.jl instead of
# re-certifying on every PR run. Expects stats_*.jld2 to already be in
# benchmark/result (run after the benchmark step, after merging splits).

stats = Dict{Symbol,DataFrame}()
load_stats("benchmark/result", stats)

reports = DataFrame[]
for key in keys(stats)
  push!(reports, certify_local_infeasibility(stats, key))
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
