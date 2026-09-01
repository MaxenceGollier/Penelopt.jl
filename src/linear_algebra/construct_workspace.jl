abstract type PenaltyWorkspace end
abstract type PenaltyDirectWorkspace <: PenaltyWorkspace end
abstract type PenaltyIterativeWorkspace <: PenaltyWorkspace end

function construct_workspace(H::M, u1::V, n::Int, m::Int; solver = :mumps) where {M,V}
  if solver == :minres_qlp
    return construct_minres_qlp_workspace(H, u1, n, m)
  elseif solver == :ldlt
    return construct_ldlt_workspace(H, u1, n, m)
  elseif solver == :ma57
    return construct_ma57_workspace(H, u1, n, m)
  elseif solver == :mumps
    return construct_mumps_workspace(H, u1, n, m)
  end
end

# HSL Misc.: Checks whether the user has the license in the extension.
const _HSL_AVAILABLE = Ref(false)
hsl_functional() = _HSL_AVAILABLE[]
_set_hsl_available(flag::Bool) = (_HSL_AVAILABLE[] = flag)

function construct_ma57_workspace(H, u1, n, m)
  error("MA57 not available. Load PeneloptHSLExt.")
end

abstract type AbstractHSLWorkspace <: PenaltyDirectWorkspace end

# Krylov Misc.
function construct_minres_qlp_workspace(H, u1, n, m)
  error("MINRES-QLP not available. Load PeneloptKrylovExt.")
end

abstract type AbstractKrylovWorkspace <: PenaltyIterativeWorkspace end

# LDLFactorizations Misc.
function construct_ldlt_workspace(H, u1, n, m)
  error("LDLFactorizations not available. Load PeneloptLDLFactorizationsExt.")
end

abstract type AbstractLDLTWorkspace <: PenaltyDirectWorkspace end

# MUMPS Misc.
abstract type AbstractMUMPSWorkspace <: PenaltyDirectWorkspace end

get_n_fact(workspace::PenaltyIterativeWorkspace) = 0
get_n_fact(workspace::PenaltyDirectWorkspace) = workspace._n_fact

function set_n_fact!(workspace::PenaltyIterativeWorkspace, n::Int)
  return
end

function set_n_fact!(workspace::PenaltyDirectWorkspace, n::Int)
  workspace._n_fact = n
end

function SolverCore.reset!(workspace::PenaltyWorkspace)
  set_n_fact!(workspace, 0)
end

# Returns whether H + σI is positive definite, where H is the current
# Hessian (of the Lagrangian) approximation stored in `workspace` and σ is
# the associated primal regularization parameter.

up_lblock_is_pos_def(workspace::PenaltyWorkspace) = up_lblock_is_pos_def(workspace, workspace.H)

function up_lblock_is_pos_def(workspace::PenaltyWorkspace, ::Any) 
  npos, nzero, nneg = get_inertia(workspace)
  return npos == workspace.n && nneg == workspace.m
end 

# A BFGS Hessian approximation is positive definite by
# construction, so H + σI is always positive definite for σ ≥ 0.
up_lblock_is_pos_def(::PenaltyWorkspace, ::CompactBFGSK2) = true

# Returns whether H + σI is singular, where H is the current Hessian (of the
# Lagrangian) approximation stored in `workspace` and σ is the associated
# primal regularization parameter.
up_lblock_is_singular(workspace::PenaltyWorkspace) = up_lblock_is_singular(workspace, workspace.H)

# TODO: implement an actual singularity check (e.g. from the inertia of the
# factorization, or a null pivot count for direct solvers). For now, we
# conservatively assume it never is, for both the BFGS and non-BFGS cases.
up_lblock_is_singular(::PenaltyWorkspace, ::CompactBFGSK2) = false
up_lblock_is_singular(::PenaltyWorkspace, ::Any) = false
