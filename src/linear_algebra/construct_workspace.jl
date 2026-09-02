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

# up_lb_is_pos_def: whether the leading ("upper-left"/primal) block H + σI of
# the augmented K2 system is positive definite. `H` is the current Hessian
# (of the Lagrangian) approximation stored in `workspace` and σ is the
# associated primal regularization parameter.
#
# The default implementations below dispatch on the type of `workspace.H`.
# Individual linear-solver workspaces (see e.g. linear_algebra/mumps.jl) may
# override these with a more expensive, but conclusive, check.

up_lb_is_pos_def(workspace::PenaltyWorkspace) = up_lb_is_pos_def(workspace, workspace.H)

# A (compact) BFGS Hessian approximation is positive definite by
# construction, so H + σI is positive definite for every σ ≥ 0.
up_lb_is_pos_def(::PenaltyWorkspace, ::CompactBFGSK2) = true

# LinearOperator-backed K2 systems (e.g. used by iterative solvers): the
# inertia of the augmented K2 system is (n, 0, m) if, and only if, H + σI is
# positive definite and A has full row rank. This is only a necessary
# condition in general (see the `Symmetric` method below and its docstring),
# but it is all we have access to without a dedicated factorization of
# H + σI on its own, so we use it as the default here.
function up_lb_is_pos_def(workspace::PenaltyWorkspace, ::AbstractLinearOperator)
  npos, nzero, nneg = get_inertia(workspace)
  return npos == workspace.n && nneg == workspace.m
end

# Direct-solver K2 systems where H is stored as a concrete (sparse) matrix:
# same necessary inertia condition as above, plus a check that all the
# diagonal entries of H + σI are positive (also necessary, but cheap).
#
# NOTE: neither of these conditions is sufficient to conclude that H + σI is
# positive definite in general -- the inertia of the K2 system only implies
# positive definiteness of H + σI via the Haynsworth inertia additivity
# formula when H + σI is itself invertible, which is exactly what we are
# trying to determine. This method is therefore kept intentionally cheap and
# "optimistic"; workspaces that can afford a conclusive check should
# override it (see PenaltyMUMPSWorkspace in linear_algebra/mumps.jl).
function up_lb_is_pos_def(workspace::PenaltyWorkspace, ::Symmetric)
  npos, nzero, nneg = get_inertia(workspace)
  (npos == workspace.n && nneg == workspace.m) || return false
  d, _ = primal_diagonal_and_row_sums(workspace)
  return all(>(0), d)
end

"""
    primal_diagonal_and_row_sums(workspace) -> (d, s)

For the sparse symmetric K2 matrix `H` stored in `workspace` (see `get_H`),
fill and return `d` and `s`, two length-`n` buffers preallocated on
`workspace` (`n = workspace.n`), such that `d[i]` is the `i`-th diagonal
entry of `H + σI`, and `s[i]` is the sum of the absolute values of the
off-diagonal entries of row `i` of `H + σI`. Only the leading n×n
("primal") block is considered; entries coupling to the constraint
Jacobian, or to the dual (-αI) block, are ignored.

`H` is assumed to store (at least) an upper-triangular view of a symmetric
matrix, possibly with repeated (row, col) entries that must be summed
(as produced e.g. by `SparseMatrixCOO`); duplicate entries in `H`, if any,
are correctly accumulated.

The returned vectors alias `workspace`'s internal buffers and are
overwritten on the next call; this function performs no allocation.
"""
function primal_diagonal_and_row_sums(workspace::PenaltyWorkspace)
  d, s = workspace._primal_diag, workspace._primal_row_sum
  fill_primal_diagonal_and_row_sums!(d, s, get_H(workspace))
  return d, s
end

function fill_primal_diagonal_and_row_sums!(
  d::AbstractVector,
  s::AbstractVector,
  H::SparseMatrixCOO,
)
  n = length(d)
  fill!(d, zero(eltype(d)))
  fill!(s, zero(eltype(s)))
  rows_, cols_, vals_ = H.rows, H.cols, H.vals
  for k in eachindex(vals_)
    i, j = rows_[k], cols_[k]
    (i > n || j > n) && continue
    v = vals_[k]
    if i == j
      d[i] += v
    else
      s[i] += abs(v)
      s[j] += abs(v)
    end
  end
  return d, s
end

function fill_primal_diagonal_and_row_sums!(
  d::AbstractVector,
  s::AbstractVector,
  H::SparseMatrixCSC,
)
  n = length(d)
  fill!(d, zero(eltype(d)))
  fill!(s, zero(eltype(s)))
  rowval, nzval = H.rowval, H.nzval
  for j = 1:n
    for k in nzrange(H, j)
      i = rowval[k]
      i > n && continue
      v = nzval[k]
      if i == j
        d[i] += v
      else
        s[i] += abs(v)
        s[j] += abs(v)
      end
    end
  end
  return d, s
end
