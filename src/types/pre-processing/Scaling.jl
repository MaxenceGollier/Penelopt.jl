"""
    scale_model.jl
 
Standalone NLP scaling wrapper for NLPModels.jl-compatible models.
 
Given scaling factors `d_f` (a scalar multiplying the objective) and `d_c`
(a vector multiplying each constraint), this wraps any `AbstractNLPModel`
so that the *scaled* problem
 
    minimize   d_f * f(x)
    subject to d_c .* lcon <= d_c .* c(x) <= d_c .* ucon
 
is exposed through the ordinary NLPModels.jl API (obj, grad!, cons!,
jac_coord!, hess_coord!, jprod!, jtprod!, hprod!, ...), so any solver that
accepts an `AbstractNLPModel` can use it directly, with no other changes.
 
Variable bounds and the initial point `x0` are left untouched: this file
only implements *objective/constraint* scaling, not variable scaling. It
composes cleanly with `remove_fixed_variables.jl` -- scale first or after,
in either order.
 
    julia> nlp = MyNLPModel(...)
    julia> nlp_free  = remove_fixed_variables(nlp)      # optional
    julia> nlp_scaled = scale_model(nlp_free, d_f, d_c)
    julia> stats = my_solver(nlp_scaled)
 
    # Map results of the scaled problem back to the original units:
    julia> f_orig  = unscale_objective(nlp_scaled, stats.objective)
    julia> c_orig  = unscale_constraints(nlp_scaled, cons(nlp_scaled, stats.solution))
    julia> y_orig  = unscale_multipliers(nlp_scaled, stats.multipliers)
    julia> x_orig  = recover_full_solution(nlp_free, stats.solution)  # if variables were removed
"""
 
using NLPModels
using LinearAlgebra
 
export ScaledModel,
  scale_model,
  unscale_objective,
  unscale_constraints,
  unscale_multipliers,
  scale_multipliers
 
"""
    ScaledModel(nlp, d_f, d_c, ...) <: AbstractNLPModel{T,S}
 
Wraps `nlp` and exposes the scaled objective `d_f * f(x)` and scaled
constraints `d_c .* c(x)`. Do not construct directly; use
[`scale_model`](@ref).
"""
struct ScaledModel{
  T,
  S<:AbstractVector{T},
  M<:AbstractNLPModel{T,S},
  Meta<:AbstractNLPModelMeta{T,S},
} <: AbstractNLPModel{T,S}
  meta::Meta
  counters::Counters
  model::M
 
  d_f::T  # objective scaling factor
  d_c::S  # constraint scaling vector, length ncon(nlp)
 
  jac_rows::Vector{Int}  # row (= constraint) index of each Jacobian nonzero, for fast scaling
  y_buffer::S             # scratch buffer, length ncon(nlp)
end
 
get_model(nlp::ScaledModel) = nlp.model
 
"""
    scale_model(nlp::AbstractNLPModel, d_f, d_c::AbstractVector)
 
Return a [`ScaledModel`](@ref) wrapping `nlp` whose objective is `d_f * f(x)`
and whose constraints are `d_c .* c(x)`. `d_f` must be a positive scalar and
`d_c` a vector of positive scaling factors, one per constraint
(`length(d_c) == get_ncon(nlp)`).
 
If `get_ncon(nlp) == 0`, `d_c` may be passed as an empty vector.
"""
function scale_model(nlp::AbstractNLPModel{T,S}; d_f::T = one(T), d_c::S = ones(get_ncon(nlp))) where {T,S}
  ncon = get_ncon(nlp)
  length(d_c) == ncon || throw(
    DimensionMismatch(
      "length(d_c) = $(length(d_c)) does not match get_ncon(nlp) = $ncon",
    ),
  )
  d_f > 0 || throw(ArgumentError("d_f must be positive, got $d_f"))
  ncon > 0 && any(d_c .<= 0) && throw(ArgumentError("every entry of d_c must be positive"))
 
  nnzj = get_nnzj(nlp.meta)
  jac_rows = Vector{Int}(undef, nnzj)
  if nnzj > 0
    jac_cols = Vector{Int}(undef, nnzj)
    NLPModels.jac_structure!(nlp, jac_rows, jac_cols)
  end
 
  lcon = ncon > 0 ? d_c .* get_lcon(nlp) : similar(d_c, 0)
  ucon = ncon > 0 ? d_c .* get_ucon(nlp) : similar(d_c, 0)
  y0 = ncon > 0 ? (get_y0(nlp) .* d_f) ./ d_c : similar(d_c, 0)
 
  meta = NLPModelMeta(
    get_nvar(nlp);
    x0 = get_x0(nlp),
    lvar = get_lvar(nlp),
    uvar = get_uvar(nlp),
    ncon = ncon,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
    nnzj = nnzj,
    nnzh = get_nnzh(nlp.meta),
    lin = nlp.meta.lin,
    minimize = nlp.meta.minimize,
    islp = false,
    name = string(nlp.meta.name, " (scaled)"),
  )
 
  return ScaledModel(meta, Counters(), nlp, d_f, d_c, jac_rows, similar(d_c))
end
 
# --- helpers to move quantities between the scaled and original problems ---
 
"""
    unscale_objective(nlp::ScaledModel, f_scaled)
 
Map an objective value of the *scaled* problem back to the original units.
"""
unscale_objective(nlp::ScaledModel, f_scaled) = f_scaled / nlp.d_f
unscale_objective(::AbstractNLPModel, f) = f  # no-op fallback
 
"""
    unscale_constraints(nlp::ScaledModel, c_scaled)
 
Map a constraint value vector of the *scaled* problem back to the original
units.
"""
unscale_constraints(nlp::ScaledModel, c_scaled::AbstractVector) = c_scaled ./ nlp.d_c
unscale_constraints(::AbstractNLPModel, c) = c  # no-op fallback
 
"""
    unscale_multipliers(nlp::ScaledModel, y_scaled)
 
Map Lagrange multipliers of the *scaled* problem back to the multipliers of
the original problem: `y = y_scaled .* d_c ./ d_f`.
"""
unscale_multipliers(nlp::ScaledModel, y_scaled::AbstractVector) = (y_scaled .* nlp.d_c) ./ nlp.d_f
unscale_multipliers(::AbstractNLPModel, y) = y  # no-op fallback
 
"""
    scale_multipliers(nlp::ScaledModel, y)
 
Map Lagrange multipliers of the *original* problem to the multipliers that
the scaled problem's stationarity condition expects:
`y_scaled = y .* d_f ./ d_c`.
"""
scale_multipliers(nlp::ScaledModel, y::AbstractVector) = (y .* nlp.d_f) ./ nlp.d_c
scale_multipliers(::AbstractNLPModel, y) = y  # no-op fallback
 
# ------------------------------------------------------------------------
# NLPModels API
# ------------------------------------------------------------------------
 
function NLPModels.obj(nlp::ScaledModel, x::AbstractVector)
  NLPModels.increment!(nlp, :neval_obj)
  return nlp.d_f * NLPModels.obj(nlp.model, x)
end
 
function NLPModels.grad!(nlp::ScaledModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(nlp, :neval_grad)
  NLPModels.grad!(nlp.model, x, g)
  g .*= nlp.d_f
  return g
end
 
function NLPModels.cons!(nlp::ScaledModel, x::AbstractVector, c::AbstractVector)
  NLPModels.increment!(nlp, :neval_cons)
  NLPModels.cons!(nlp.model, x, c)
  c .*= nlp.d_c
  return c
end
 
function NLPModels.jac_structure!(nlp::ScaledModel, rows::AbstractVector, cols::AbstractVector)
  NLPModels.jac_structure!(nlp.model, rows, cols)
  return rows, cols
end
 
function NLPModels.jac_coord!(nlp::ScaledModel, x::AbstractVector, vals::AbstractVector)
  NLPModels.increment!(nlp, :neval_jac)
  NLPModels.jac_coord!(nlp.model, x, vals)
  @inbounds for k in eachindex(vals)
    vals[k] *= nlp.d_c[nlp.jac_rows[k]]
  end
  return vals
end
 
function NLPModels.jprod!(
  nlp::ScaledModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  NLPModels.increment!(nlp, :neval_jprod)
  NLPModels.jprod!(nlp.model, x, v, Jv)
  Jv .*= nlp.d_c
  return Jv
end
 
function NLPModels.jtprod!(
  nlp::ScaledModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  NLPModels.increment!(nlp, :neval_jtprod)
  nlp.y_buffer .= v .* nlp.d_c
  NLPModels.jtprod!(nlp.model, x, nlp.y_buffer, Jtv)
  return Jtv
end
 
function NLPModels.hess_structure!(nlp::ScaledModel, rows::AbstractVector, cols::AbstractVector)
  NLPModels.hess_structure!(nlp.model, rows, cols)
  return rows, cols
end
 
# Objective-only Hessian (e.g. unconstrained or Gauss-Newton style calls)
function NLPModels.hess_coord!(
  nlp::ScaledModel{T},
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight = one(T),
) where {T}
  NLPModels.increment!(nlp, :neval_hess)
  NLPModels.hess_coord!(nlp.model, x, vals; obj_weight = obj_weight * nlp.d_f)
  return vals
end
 
# Hessian of the Lagrangian: y here is the multiplier of the *scaled*
# constraints, so we convert it back to what the underlying (unscaled)
# model expects before delegating.
function NLPModels.hess_coord!(
  nlp::ScaledModel{T},
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight = one(T),
) where {T}
  NLPModels.increment!(nlp, :neval_hess)
  nlp.y_buffer .= y .* nlp.d_c
  NLPModels.hess_coord!(nlp.model, x, nlp.y_buffer, vals; obj_weight = obj_weight * nlp.d_f)
  return vals
end
 
function NLPModels.hprod!(
  nlp::ScaledModel{T},
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(T),
) where {T}
  NLPModels.increment!(nlp, :neval_hprod)
  NLPModels.hprod!(nlp.model, x, v, Hv; obj_weight = obj_weight * nlp.d_f)
  return Hv
end
 
function NLPModels.hprod!(
  nlp::ScaledModel{T},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(T),
) where {T}
  NLPModels.increment!(nlp, :neval_hprod)
  nlp.y_buffer .= y .* nlp.d_c
  NLPModels.hprod!(nlp.model, x, nlp.y_buffer, v, Hv; obj_weight = obj_weight * nlp.d_f)
  return Hv
end
