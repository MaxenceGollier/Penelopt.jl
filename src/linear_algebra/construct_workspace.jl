abstract type PenaltyWorkspace end
abstract type PenaltyDirectWorkspace <: PenaltyWorkspace end
abstract type PenaltyIterativeWorkspace <: PenaltyWorkspace end

function construct_workspace(H::M, u1::V, n::Int, m::Int; solver = :minres_qlp) where {M,V}
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
  error("MA57 not available. Load ExactPenaltyHSLExt.")
end

abstract type AbstractHSLWorkspace <: PenaltyDirectWorkspace end

# Krylov Misc.
function construct_minres_qlp_workspace(H, u1, n, m)
  error("MINRES-QLP not available. Load ExactPenaltyKrylovExt.")
end

abstract type AbstractKrylovWorkspace <: PenaltyIterativeWorkspace end

# MUMPS Misc.
function construct_mumps_workspace(H, u1, n, m)
  error("MUMPS not available. Load ExactPenaltyMUMPSExt.")
end

abstract type AbstractMUMPSWorkspace <: PenaltyDirectWorkspace end

get_n_fact(workspace::PenaltyIterativeWorkspace) = 0
get_n_fact(workspace::PenaltyDirectWorkspace) = workspace._n_fact

function set_n_fact!(workspace::PenaltyIterativeWorkspace, n::Int)
  return
end

function set_n_fact!(workspace::PenaltyDirectWorkspace, n::Int)
  workspace._n_fact = n
end
