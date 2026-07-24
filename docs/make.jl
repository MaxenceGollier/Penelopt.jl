using ExactPenalty
using Documenter

DocMeta.setdocmeta!(ExactPenalty, :DocTestSetup, :(using ExactPenalty); recursive = true)

makedocs(;
  modules = [ExactPenalty],
  authors = "Maxence Gollier maxence-2.gollier@polymtl.ca",
  repo = "https://github.com/MaxenceGollier/ExactPenalty.jl/blob/{commit}{path}#{line}",
  sitename = "ExactPenalty.jl",
  format = Documenter.HTML(;
    canonical = "https://MaxenceGollier.github.io/ExactPenalty.jl",
    assets = ["assets/link-icons.css"],
    collapselevel = 1
  ),
  workdir = joinpath(@__DIR__, "src"),
  pages = [
    "Home" => "index.md",
    "Options" => "options.md",
    "Outputs" => "outputs.md",
    "Performance" => "performance.md",
    "Callbacks" => "callbacks.md",
    "Tutorials" => [
      "AMPL" => "tutorials/AMPL.md",
      "CUTEst" => "tutorials/CUTEst.md",
      "JuMP" => "tutorials/JuMP.md",
      "HSL" => "tutorials/HSL.md",
      "MUMPS" => "tutorials/MUMPS.md",
    ],
    "Developers" => [
      "Contributing" => "90-contributing.md",
      "Developing" => "91-developer.md",
    ]
  ],
)

deploydocs(; repo = "github.com/MaxenceGollier/ExactPenalty.jl", push_preview = true)
