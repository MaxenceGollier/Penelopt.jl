mutable struct PenaltyMUMPSWorkspace{
  WP<:Mumps,
  K2<:AbstractMatrix,
  V<:AbstractVector,
  T<:Real,
} <: AbstractMUMPSWorkspace
  M::WP
  H::K2
  x::V
  σ::T
  n::Int
  m::Int
  status::Symbol
  factorized::Bool
  _n_fact::Int
end

function get_H(
  solver_workspace::PenaltyMUMPSWorkspace{WP,K2},
) where {T,M,WP,K2<:Symmetric{T,M}}
  return solver_workspace.H.data
end

function get_H(solver_workspace::PenaltyMUMPSWorkspace{WP,K2}) where {WP,K2<:CompactBFGSK2}
  return solver_workspace.H.H.data
end

function construct_mumps_workspace(
  H::M,
  u1::V,
  n,
  m;
  max_m_lapack::Int = 100,
) where {T,V<:AbstractVector{T},M<:Symmetric}
  # When there are few enough constraints, it is cheaper to have MUMPS
  # factor only the large n×n leading (Hessian) block, retrieve the small
  # dense m×m Schur complement it assembles along the way, and factor that
  # ourselves with LAPACK (Bunch-Kaufman) — rather than factoring the full
  # (n+m)×(n+m) augmented system with MUMPS every time.
  if m <= max_m_lapack
    return construct_mumps_schur_lapack_workspace(H, u1, n, m)
  end

  # Set params : TODO
  cntl = T == Float64 ? default_cntl64 : default_cntl32
  icntl = default_icntl

  ## Set Parameters

  # CNTL(1) is the relative threshold for numerical pivoting.
  # Remarks: It forms a trade-off between preserving sparsity and ensuring numerical stability during
  # the factorization. In general, a larger value of CNTL(1) increases fill-in but leads to a more accurate
  # factorization.
  cntl[1] = eps(T)

  cntl[2] = eps(T) # Tolerance for iterative refinement

  # Deactivate Logging
  icntl[2], icntl[3], icntl[4] = 0, 0, 0

  # Max number of iterative refinement steps
  icntl[10] = 10

  # ICNTL(11): error analysis
  # 2: Main statistics (recommended)
  icntl[11] = 2

  # ICNTL(24) controls the detection of “null pivot rows”.
  # 1: Null pivot row detection.
  icntl[24] = 1

  # CNTL(13) controls the parallelism of the root node
  # Remarks: Processing the root sequentially (ICNTL(13) > 0) can be useful when the user is
  # interested in the inertia of the matrix (see INFO(12) and INFOG(12)), or when the user wants
  # to detect null pivots (see Subsection 5.13) or to activate BLR compression (Subsection 5.20) on the
  # root node.
  icntl[13] = 1

  S = Mumps{T}(mumps_symmetric, icntl, cntl)

  # Associate the row, cols and vals of the mumps structure with those of H.
  irn, jcn, a = H.data.rows, H.data.cols, H.data.vals
  S.irn, S.jcn, S.a = pointer.((irn, jcn, a))
  S.n = m+n
  S.nnz = length(irn)
  S._irn_gc_haven = irn
  S._jcn_gc_haven = jcn
  S._a_gc_haven = a

  # Associate the size and number of the right hand side
  x = similar(u1)
  S.lrhs = n + m
  S.nrhs = 1
  S.rhs = pointer(x)
  S._y_gc_haven = x

  return PenaltyMUMPSWorkspace(S, H, x, zero(T), n, m, :uninitialized, false, 0)
end

function construct_mumps_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M<:CompactBFGSK2}
  # Set params : TODO
  cntl = T == Float64 ? default_cntl64 : default_cntl32
  icntl = default_icntl

  ## Set Parameters
  # CNTL(1) is the relative threshold for numerical pivoting.
  # Remarks: It forms a trade-off between preserving sparsity and ensuring numerical stability during
  # the factorization. In general, a larger value of CNTL(1) increases fill-in but leads to a more accurate
  # factorization.
  cntl[1] = eps(T)

  cntl[2] = eps(T) # Tolerance for iterative refinement

  # Deactivate Logging
  icntl[2], icntl[3], icntl[4] = 0, 0, 0

  # Max number of iterative refinement steps
  icntl[10] = 10

  # ICNTL(11): error analysis
  # 2: Main statistics (recommended)
  icntl[11] = 2

  # CNTL(13) controls the parallelism of the root node
  # Remarks: Processing the root sequentially (ICNTL(13) > 0) can be useful when the user is
  # interested in the inertia of the matrix (see INFO(12) and INFOG(12)), or when the user wants
  # to detect null pivots (see Subsection 5.13) or to activate BLR compression (Subsection 5.20) on the
  # root node.
  icntl[13] = 1

  # ICNTL(24) controls the detection of “null pivot rows”.
  # 1: Null pivot row detection.
  icntl[24] = 1

  S = Mumps{T}(mumps_symmetric, icntl, cntl)

  # Associate the row, cols and vals of the mumps structure with those of H.
  irn, jcn, a = H.H.data.rows, H.H.data.cols, H.H.data.vals
  S.irn, S.jcn, S.a = pointer.((irn, jcn, a))
  S.n = m+n
  S.nnz = length(irn)
  S._irn_gc_haven = irn
  S._jcn_gc_haven = jcn
  S._a_gc_haven = a

  # Associate the size and number of the right hand side
  x = similar(u1)
  S.lrhs = n + m
  S.nrhs = 1
  S.rhs = pointer(x)
  S._y_gc_haven = x

  return PenaltyMUMPSWorkspace(S, H, x, zero(T), n, m, :uninitialized, false, 0)
end

function update_workspace!(
  solver_workspace::PenaltyMUMPSWorkspace,
  B::M,
  A,
  σ,
  α,
) where {M<:SparseMatrixCOO}
  n, m = solver_workspace.n, solver_workspace.m
  nnz_B, nnz_A = length(B.vals), length(A.vals)

  H = get_H(solver_workspace)

  H.vals[1:nnz_B] .= B.vals
  H.vals[(nnz_B+1):(nnz_B+nnz_A)] .= A.vals
  H.vals[(nnz_B+nnz_A+1):(nnz_B+nnz_A+n)] .= σ
  H.vals[(nnz_B+nnz_A+n+1):(nnz_B+nnz_A+n+m)] .= -α

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function update_workspace!(solver_workspace::PenaltyMUMPSWorkspace, A, σ, α)
  # Warning: Considers tht B is a zero matrix.
  n, m = solver_workspace.n, solver_workspace.m
  nnz_A = length(A.vals)
  H = get_H(solver_workspace)
  nnz_B = length(H.vals) - nnz_A - n - m

  H.vals .= 0
  H.vals[(nnz_B+1):(nnz_B+nnz_A)] .= A.vals
  H.vals[(nnz_B+nnz_A+1):(nnz_B+nnz_A+n)] .= σ
  H.vals[(nnz_B+nnz_A+n+1):(nnz_B+nnz_A+n+m)] .= -α

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function update_workspace!(
  solver_workspace::PenaltyMUMPSWorkspace,
  B::M,
  A,
  σ,
  α,
) where {M<:CompactBFGS}
  n, m = solver_workspace.n, solver_workspace.m
  nnz_A = length(A.vals)

  H = get_H(solver_workspace)

  H.vals[1:nnz_A] .= A.vals
  H.vals[(nnz_A+1):(nnz_A+n)] .= σ + B.ξ
  H.vals[(nnz_A+n+1):(nnz_A+n+m)] .= -α

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function set_dual_inertia!(solver_workspace::PenaltyMUMPSWorkspace, α)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  H.vals[(end-m+1):end] .= -α
  solver_workspace.factorized = false
end

function set_primal_inertia!(solver_workspace::PenaltyMUMPSWorkspace, σ)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  H.vals[(end-m-n+1):(end-m)] .= σ
  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function solve_system!(
  workspace::PenaltyMUMPSWorkspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2}
  workspace.status = :success
  mumps, H = workspace.M, workspace.H

  if !workspace.factorized
    job = mumps.job
    mumps.job = MUMPS.INITIALIZE
    factorize!(mumps)
    workspace._n_fact += 1

    k, max_iter = 0, 5
    # MUMPS Documentation - infog(1) = -9
    # The main internal real/complex workarray S is too small. If INFO(2) is positive, then the number
    # of entries that are missing in S at the moment when the error is raised is available in INFO(2).
    # If INFO(2) is negative, then its absolute value should be multiplied by 1 million. If an error –9
    # occurs, the user should increase the value of ICNTL(14) before calling the factorization (JOB=
    # 2) again, except if LWK USER is provided LWK USER should be increased.
    while mumps.infog[1] == -9 && k < max_iter
      MUMPS.set_icntl!(mumps, 14, mumps.icntl[14] * 2)
      mumps.job = MUMPS.FACTOR
      factorize!(mumps)
      workspace._n_fact += 1
      k = k + 1
    end

    # MUMPS infog(1): a negative value is an error in the factorization.
    if mumps.infog[1] < 0
      workspace.status = :failed
      return
    else
      workspace.factorized = true
    end
  end

  workspace.x .= u
  MUMPS.mumps_solve!(workspace.x, mumps; rhs_changed = true)

  # MUMPS infog(1): a negative value is an error in the factorization.
  if any(isnan, workspace.x) || mumps.infog[1] < 0
    workspace.status = :failed
  end

  update_pivtol!(workspace)

  return
end

function solve_system!(
  workspace::PenaltyMUMPSWorkspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2<:CompactBFGSK2}
  workspace.status = :success
  H = workspace.H
  B = workspace.H.B
  mumps = workspace.M
  n, m = workspace.n, workspace.m
  p = min(B._insert - 1, B._mem)
  x1, x2, x3, y1, y2 = H.x1, H.x2, H.x3, H.y1, H.y2
  Z1, Z2 = H.Z1, H.Z2

  Uk = @view B.Uk[:, 1:p]
  Vk = @view B.Vk[:, 1:p]

  # Step 0: Write
  # [B  Aᵀ] = [σI+ξI  Aᵀ] + [-U V]([U V])ᵀ
  # [A -αI] = [A     -αI] + [ 0 0]([0 0])
  # Hence,
  # [B  Aᵀ] = [σI+ξI  Aᵀ] + EFᵀ
  # [A -αI] = [A     -αI] + EFᵀ

  # Step 1: Factorize
  # [σI+ξI  Aᵀ]
  # [A     -αI]

  if !workspace.factorized
    job = mumps.job
    mumps.job = MUMPS.INITIALIZE
    factorize!(mumps)
    workspace._n_fact += 1

    k, max_iter = 0, 5
    # MUMPS Documentation - infog(1) = -9
    # The main internal real/complex workarray S is too small. If INFO(2) is positive, then the number
    # of entries that are missing in S at the moment when the error is raised is available in INFO(2).
    # If INFO(2) is negative, then its absolute value should be multiplied by 1 million. If an error –9
    # occurs, the user should increase the value of ICNTL(14) before calling the factorization (JOB=
    # 2) again, except if LWK USER is provided LWK USER should be increased.
    while mumps.infog[1] == -9 && k < max_iter
      MUMPS.set_icntl!(mumps, 14, mumps.icntl[14] * 2)
      mumps.job = MUMPS.FACTOR
      factorize!(mumps)
      workspace._n_fact += 1
      k = k + 1
    end

    # MUMPS infog(1): a negative value is an error in the factorization.
    if mumps.infog[1] < 0
      workspace.status = :failed
      return
    else
      workspace.factorized = true
    end
  end

  # Step 2: Compute
  # [x₁] = [σI+ξI  Aᵀ]⁻¹[u]
  # [x₁] = [A     -αI]  [u]

  x1 .= u
  MUMPS.associate_rhs!(mumps, x1)
  MUMPS.mumps_solve!(x1, mumps; rhs_changed = true)

  # MUMPS infog(1): a negative value is an error in the factorization.
  if any(isnan, x1) || mumps.infog[1] < 0
    workspace.status = :failed
    return
  end

  update_pivtol!(workspace)

  # Step 3: Compute
  # y₁ = Fᵀx₁ = [Uᵀx₁(1:n)]
  # y₁ = Fᵀx₁ = [Vᵀx₁(1:n)]
  @views mul!(y1[1:p], Uk', x1[1:n])
  @views mul!(y1[(p+1):(2*p)], Vk', x1[1:n])


  # Step 4: Assemble Schur complement (I + Fᵀ [σI+ξI  Aᵀ]⁻¹ E )
  #                                   (       [A     -αI]     )
  # Step 4.1: Compute 
  # Z₁ = [σI+ξI  Aᵀ]⁻¹ E = [σI+ξI  Aᵀ]⁻¹[-U V]
  # Z₁ = [A     -αI]   E = [A     -αI]  [ 0 0]
  Z1 .= 0

  @views Z1[1:n, 1:p] .= Uk .* (-1)
  @views Z1[1:n, (p+1):(2*p)] .= Vk

  MUMPS.associate_rhs!(mumps, Z1)
  MUMPS.mumps_solve!(Z1, mumps; rhs_changed = true)

  # MUMPS infog(1): a negative value is an error in the factorization.
  if any(isnan, Z1) || mumps.infog[1] < 0
    workspace.status = :failed
    return
  end

  update_pivtol!(workspace)

  # Step 4.2: Compute 
  # Z₂ = FᵀZ₁ = UᵀZ₁[1:n]
  # Z₂ = FᵀZ₁ = VᵀZ₁[1:n]
  Z2 .= 0
  @views mul!(Z2[1:p, 1:(2*p)], Uk', Z1[1:n, (1:(2*p))])
  @views mul!(Z2[(p+1):(2*p), 1:(2*p)], Vk', Z1[1:n, (1:(2*p))])

  # Step 4.3: Compute 
  # Z₂ = I + Z₂
  for i = 1:(2*p)
    Z2[i, i] += 1
  end

  # Step 5: Solve
  # (I + Fᵀ [σI+ξI  Aᵀ]⁻¹ E )⁻¹[y₁]
  # (       [A     -αI]     )  [y₁]
  # using Julia LinearALgebra's lu!
  F = lu!(Z2[1:(2*p), 1:(2*p)], check = false) # FIXME ?
  @views ldiv!(y2[1:(2*p)], F, y1[1:(2*p)])
  if any(isnan, y2)
    workspace.status = :failed
    return
  end

  # Step 6: Compute
  # x₂ = E[y₂] = [-U V][y₂] = [-Uy₂ + Vy₂]
  # x₂ = E[y₂] = [ 0 0][y₂] = [0]
  @views mul!(x2[1:n], Vk, y2[(p+1):(2*p)])
  @views mul!(x2[1:n], Uk, y2[1:p], -one(eltype(y2)), one(eltype(y2)))

  # Step 7: Solve
  # [x₃] = [σI+ξI  Aᵀ]⁻¹[x₂]
  # [x₃] = [A     -αI]  [x₂]
  x3 .= x2
  MUMPS.associate_rhs!(mumps, x3)
  MUMPS.mumps_solve!(x3, mumps; rhs_changed = true)

  # MUMPS infog(1): a negative value is an error in the factorization.
  if any(isnan, x3) || mumps.infog[1] < 0
    workspace.status = :failed
    return
  end

  update_pivtol!(workspace)

  # Step 8:
  # [B  Aᵀ]⁻¹[u] = x₁ - x₃ 
  # [A -αI]  [u] = x₁ - x₃
  workspace.x .= x1 .- x3
end

function get_solution!(x::V, workspace::AbstractMUMPSWorkspace) where {V<:AbstractVector}
  x .= workspace.x
end

function get_status(workspace::AbstractMUMPSWorkspace)
  return workspace.status
end

function get_inertia(workspace::PenaltyMUMPSWorkspace{WP,K2}) where {WP,K2}

  n, m = workspace.n, workspace.m
  (npos, nzero, nneg) = (0, 0, 0)

  nneg = workspace.M.infog[12]
  rank = n + m - workspace.M.infog[28]
  nzero = n + m - rank
  npos = n + m - nzero - nneg

  return npos, nzero, nneg
end

function relative_error!(workspace::AbstractMUMPSWorkspace)
  mumps = workspace.M

  return max(mumps.rinfog[6], mumps.rinfog[7], mumps.rinfog[8])
end

function increase_pivtol!(workspace::AbstractMUMPSWorkspace)
  mumps = workspace.M

  MUMPS.set_cntl!(mumps, 1, 1e-2)
  MUMPS.set_icntl!(mumps, 10, -10)
end

function decrease_pivtol!(workspace::AbstractMUMPSWorkspace)
  mumps = workspace.M

  MUMPS.set_cntl!(mumps, 1, mumps.cntl[1] / 10)
  MUMPS.set_icntl!(mumps, 10, 10)
end

function update_pivtol!(workspace::AbstractMUMPSWorkspace)
  mumps = workspace.M

  relative_error = relative_error!(workspace)
  if relative_error > sqrt(eps(eltype(workspace.x)))
    increase_pivtol!(workspace)
  elseif relative_error < eps(eltype(workspace.x)) / 100
    decrease_pivtol!(workspace)
  end
end

function SolverCore.reset!(workspace::PenaltyMUMPSWorkspace)
  set_n_fact!(workspace, 0)
  MUMPS.set_icntl!(workspace.M, 10, 10)
  MUMPS.set_cntl!(workspace.M, 1, eps(eltype(workspace.x)))
end

# MUMPS + Schur complement + LAPACK Misc.
#
# Used instead of `PenaltyMUMPSWorkspace` when the number of constraints m
# is small (m <= max_m_lapack, see `construct_mumps_workspace`). Rather than
# have MUMPS factor the full (n+m)×(n+m) K2 system
#   [ B+σI  Aᵀ ]
#   [ A    -αI ]
# every time, we designate the last m (dual/constraint) indices as Schur
# variables: MUMPS factors only the large n×n leading block and hands back
# the small, dense m×m Schur complement
#   S = -αI - A(B+σI)⁻¹Aᵀ,
# which we factor ourselves with a Bunch-Kaufman (LAPACK) factorization.
# Solves then use MUMPS's own reduced-RHS mechanism (ICNTL(19)=3 /
# ICNTL(26)) to do the forward/backward elimination around the large block,
# with the small system in between solved via our LAPACK factors. This is
# not used for the CompactBFGS case, which already has its own cheap
# Sherman-Morrison-based `solve_system!`.
mutable struct PenaltyMUMPSSchurLAPACKWorkspace{
  WP<:Mumps,
  K2<:AbstractMatrix,
  V<:AbstractVector,
  T<:Real,
} <: AbstractMUMPSWorkspace
  M::WP
  H::K2
  x::V           # full (n+m) solution/rhs buffer, registered with MUMPS once
  σ::T
  n::Int
  m::Int
  S::Matrix{T}   # dense m×m Schur complement, factored in place (uplo = 'U')
  _redrhs::Vector{T}     # MUMPS reduced-RHS buffer (length m), registered once
  _ipiv::Vector{BlasInt}  # Bunch-Kaufman pivots for S
  _work::Vector{T}        # LAPACK workspace for sytrf!
  _info::Base.RefValue{BlasInt}
  status::Symbol
  factorized::Bool
  _n_fact::Int
end

function get_H(
  solver_workspace::PenaltyMUMPSSchurLAPACKWorkspace{WP,K2},
) where {T,M,WP,K2<:Symmetric{T,M}}
  return solver_workspace.H.data
end

function construct_mumps_schur_lapack_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M<:Symmetric}
  cntl = T == Float64 ? default_cntl64 : default_cntl32
  icntl = default_icntl

  cntl[1] = eps(T)
  cntl[2] = eps(T) # Tolerance for iterative refinement

  # Deactivate Logging
  icntl[2], icntl[3], icntl[4] = 0, 0, 0

  # Max number of iterative refinement steps
  icntl[10] = 10

  # ICNTL(11): error analysis
  icntl[11] = 2

  # CNTL(13): process the root sequentially, needed to read off the
  # inertia of the leading block from INFOG(12)/INFOG(28) below.
  icntl[13] = 1

  # ICNTL(24): null pivot row detection.
  icntl[24] = 1

  # ICNTL(8): turn off scaling — not supported together with the Schur
  # complement (MUMPS would otherwise warn on every factorization).
  icntl[8] = 0

  S = Mumps{T}(mumps_symmetric, icntl, cntl)

  # Associate the row, cols and vals of the mumps structure with those of H.
  irn, jcn, a = H.data.rows, H.data.cols, H.data.vals
  S.irn, S.jcn, S.a = pointer.((irn, jcn, a))
  S.n = m + n
  S.nnz = length(irn)
  S._irn_gc_haven = irn
  S._jcn_gc_haven = jcn
  S._a_gc_haven = a

  # Designate the m dual (constraint) variables — the last m indices of the
  # augmented K2 system — as Schur variables, so MUMPS eliminates the
  # leading n×n block internally and returns the dense m×m Schur
  # complement.
  schur_inds = collect((n+1):(n+m))
  MUMPS.set_schur_centralized_by_column!(S, schur_inds)

  # Associate the size and number of the right hand side. This buffer is
  # registered with MUMPS once and reused (in place) for every solve.
  x = similar(u1)
  S.lrhs = n + m
  S.nrhs = 1
  S.rhs = pointer(x)
  S._y_gc_haven = x

  # MUMPS's reduced-RHS buffer for the Schur (dual) block, also registered
  # once and reused across solves.
  redrhs = zeros(T, m)
  S.redrhs = pointer(redrhs)
  S.lredrhs = m

  Sd = Matrix{T}(undef, m, m)
  ipiv = Vector{BlasInt}(undef, m)
  # A generously-sized fixed workspace for sytrf! (m is small — at most
  # max_m_lapack — so this is cheap): avoids the extra ccall needed for a
  # proper LWORK query.
  work = Vector{T}(undef, max(1, 64 * m))

  return PenaltyMUMPSSchurLAPACKWorkspace(
    S,
    H,
    x,
    zero(T),
    n,
    m,
    Sd,
    redrhs,
    ipiv,
    work,
    Ref{BlasInt}(0),
    :uninitialized,
    false,
    0,
  )
end

function update_workspace!(
  solver_workspace::PenaltyMUMPSSchurLAPACKWorkspace,
  B::M,
  A,
  σ,
  α,
) where {M<:SparseMatrixCOO}
  n, m = solver_workspace.n, solver_workspace.m
  nnz_B, nnz_A = length(B.vals), length(A.vals)

  H = get_H(solver_workspace)

  H.vals[1:nnz_B] .= B.vals
  H.vals[(nnz_B+1):(nnz_B+nnz_A)] .= A.vals
  H.vals[(nnz_B+nnz_A+1):(nnz_B+nnz_A+n)] .= σ
  H.vals[(nnz_B+nnz_A+n+1):(nnz_B+nnz_A+n+m)] .= -α

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function update_workspace!(solver_workspace::PenaltyMUMPSSchurLAPACKWorkspace, A, σ, α)
  # Warning: Considers that B is a zero matrix.
  n, m = solver_workspace.n, solver_workspace.m
  nnz_A = length(A.vals)
  H = get_H(solver_workspace)
  nnz_B = length(H.vals) - nnz_A - n - m

  H.vals .= 0
  H.vals[(nnz_B+1):(nnz_B+nnz_A)] .= A.vals
  H.vals[(nnz_B+nnz_A+1):(nnz_B+nnz_A+n)] .= σ
  H.vals[(nnz_B+nnz_A+n+1):(nnz_B+nnz_A+n+m)] .= -α

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function set_dual_inertia!(solver_workspace::PenaltyMUMPSSchurLAPACKWorkspace, α)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  H.vals[(end-m+1):end] .= -α
  solver_workspace.factorized = false
end

function set_primal_inertia!(solver_workspace::PenaltyMUMPSSchurLAPACKWorkspace, σ)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  H.vals[(end-m-n+1):(end-m)] .= σ
  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function solve_system!(
  workspace::PenaltyMUMPSSchurLAPACKWorkspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2}
  workspace.status = :success
  mumps = workspace.M
  n, m = workspace.n, workspace.m

  if !workspace.factorized
    mumps.job = MUMPS.INITIALIZE
    factorize!(mumps)
    workspace._n_fact += 1

    k, max_iter = 0, 5
    # See the plain PenaltyMUMPSWorkspace solve_system! for context on
    # infog(1) == -9 (workarray too small) retries.
    while mumps.infog[1] == -9 && k < max_iter
      MUMPS.set_icntl!(mumps, 14, mumps.icntl[14] * 2)
      mumps.job = MUMPS.FACTOR
      factorize!(mumps)
      workspace._n_fact += 1
      k = k + 1
    end

    if mumps.infog[1] < 0
      workspace.status = :failed
      return
    else
      workspace.factorized = true
    end

    # Retrieve the dense m×m Schur complement MUMPS assembled while
    # eliminating the leading n×n block, and Bunch-Kaufman-factor it so we
    # can both solve against it and read off its inertia (see
    # `get_inertia` below).
    MUMPS.get_schur_complement!(workspace.S, mumps)

    info_f = sytrf!(
      'U',
      BlasInt(m),
      workspace.S,
      BlasInt(m),
      workspace._ipiv,
      workspace._work,
      BlasInt(length(workspace._work)),
      workspace._info,
    )
    if info_f < 0
      # A negative info indicates an illegal ccall argument: a bug, not a
      # numerical failure.
      workspace.status = :failed
      return
    end
    # info_f > 0 (an exactly-zero pivot) is not treated as a hard failure
    # here: it is picked up as a zero eigenvalue by `get_inertia` instead.
  end

  # Forward step: partial solve to obtain the reduced RHS on the Schur
  # (dual) block. `workspace.x` is the same buffer registered with MUMPS
  # at construction time, so writing into it in place is enough to update
  # the RHS MUMPS sees.
  workspace.x .= u
  MUMPS.set_icntl!(mumps, 26, 1; displaylevel = 0)
  mumps.job = MUMPS.SOLVE
  MUMPS.invoke_mumps!(mumps)

  if mumps.infog[1] < 0
    workspace.status = :failed
    return
  end

  # Solve the small dense Schur system in place, using the Bunch-Kaufman
  # factors computed above.
  info_s = sytrs!(
    'U',
    BlasInt(m),
    BlasInt(1),
    workspace.S,
    BlasInt(m),
    workspace._ipiv,
    workspace._redrhs,
    BlasInt(m),
    workspace._info,
  )
  if info_s != 0 || any(isnan, workspace._redrhs)
    workspace.status = :failed
    return
  end

  # Backward step: expand the full (n+m) solution from the (now solved)
  # reduced RHS, written back into `workspace.x`.
  MUMPS.set_icntl!(mumps, 26, 2; displaylevel = 0)
  mumps.job = MUMPS.SOLVE
  MUMPS.invoke_mumps!(mumps)

  if any(isnan, workspace.x) || mumps.infog[1] < 0
    workspace.status = :failed
  end

  update_pivtol!(workspace)

  return
end

# Combines, via Sylvester's law of inertia, the inertia of the leading n×n
# block (eliminated by MUMPS, read off INFOG(12)/INFOG(28)) with the
# inertia of the dense m×m Schur complement (from our Bunch-Kaufman
# factorization): inertia(K2) = inertia(leading block) + inertia(schur
# complement).
function get_inertia(workspace::PenaltyMUMPSSchurLAPACKWorkspace)
  n, m = workspace.n, workspace.m
  mumps = workspace.M

  nneg_block = mumps.infog[12]
  rank_block = n - mumps.infog[28]
  nzero_block = n - rank_block
  npos_block = n - nzero_block - nneg_block

  npos_s, nzero_s, nneg_s = bunchkaufman_inertia(workspace.S, workspace._ipiv, m; uplo = 'U')

  return npos_block + npos_s, nzero_block + nzero_s, nneg_block + nneg_s
end

# `up_lblock_is_pos_def`'s generic (::PenaltyWorkspace, ::Any) fallback
# already gives the right answer here (it calls `get_inertia`, which now
# dispatches to the method above). `up_lblock_is_singular` still needs its
# own override, following the same "correct full-system inertia" check.
up_lblock_is_singular(workspace::PenaltyMUMPSSchurLAPACKWorkspace, ::AbstractMatrix) =
  get_inertia(workspace)[2] > 0

function SolverCore.reset!(workspace::PenaltyMUMPSSchurLAPACKWorkspace)
  set_n_fact!(workspace, 0)
  MUMPS.set_icntl!(workspace.M, 10, 10)
  MUMPS.set_cntl!(workspace.M, 1, eps(eltype(workspace.x)))
end
