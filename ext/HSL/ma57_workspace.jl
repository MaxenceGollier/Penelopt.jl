mutable struct PenaltyMA57Workspace{
  WP<:Ma57,
  K2<:AbstractMatrix,
  V<:AbstractVector,
  T<:Real,
} <: AbstractHSLWorkspace
  M::WP
  H::K2
  x::V
  work::V
  _qn_work::V
  dx::V
  σ::T
  n::Int
  m::Int
  status::Symbol
  factorized::Bool
  _n_fact::Int
end

function get_H(
  solver_workspace::PenaltyMA57Workspace{WP,K2},
) where {T,WP,K2<:Symmetric{T,SparseMatrixCOO{T,Int}}}
  return solver_workspace.H.data
end

function get_H(solver_workspace::PenaltyMA57Workspace{WP,K2}) where {WP,K2<:CompactBFGSK2}
  return solver_workspace.H.H.data
end

function construct_ma57_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M<:Symmetric{T,SparseMatrixCOO{T,Int}}}
  S = ma57_coord(n + m, H.data.rows, H.data.cols, H.data.vals, sqd = true, print_level = -1)
  return PenaltyMA57Workspace(
    S,
    H,
    similar(u1),
    similar(u1, 4*length(u1)),
    similar(u1, 0),
    similar(u1),
    zero(T),
    n,
    m,
    :uninitialized,
    false,
    0,
  )
end

function construct_ma57_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M<:CompactBFGSK2}
  S = ma57_coord(
    n + m,
    H.H.data.rows,
    H.H.data.cols,
    H.H.data.vals,
    sqd = true,
    print_level = -1,
  )
  return PenaltyMA57Workspace(
    S,
    H,
    similar(u1),
    similar(u1, 4*length(u1)),
    similar(u1, 2*length(u1)*H.B._mem),
    similar(u1),
    zero(T),
    n,
    m,
    :uninitialized,
    false,
    0,
  )
end

function update_workspace!(
  solver_workspace::PenaltyMA57Workspace,
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

  solver_workspace.M.vals .= H.vals

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function update_workspace!(
  solver_workspace::PenaltyMA57Workspace,
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

  solver_workspace.M.vals .= H.vals

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function set_dual_inertia!(solver_workspace::PenaltyMA57Workspace, α)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  H.vals[(end-m+1):end] .= -α
  solver_workspace.M.vals[(end-m+1):end] .= -α
  solver_workspace.factorized = false
end

function set_primal_inertia!(solver_workspace::PenaltyMA57Workspace, σ)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  H.vals[(end-m-n+1):(end-m)] .= σ
  solver_workspace.M.vals[(end-m-n+1):(end-m)] .= σ
  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function solve_system!(
  workspace::PenaltyMA57Workspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2}
  workspace.status = :success

  # Ma57 icntl(7): Controls the pivotting strategy.
  # Ma57 icntl(7): A value of 3 performs no pivoting: for a sufficiently large value of σ and a positive value of α,
  # The matrix is strongly factorizable, so we don't need pivoting.
  workspace.M.control.icntl[7] = 3

  if !workspace.factorized
    try
      ma57_factorize!(workspace.M)
    catch e
      !(e isa HSL.Ma57Exception) && rethrow(e)
    end
    workspace._n_fact += 1
  end

  # Ma57 info(1): a negative value is an error in the factorization.
  # Ma57 info(1): a value of 4 indicates that the matrix is singular.
  if workspace.M.info.info[1] < 0 || workspace.M.info.info[1] == 4
    workspace.status = :failed
    return
  else
    workspace.factorized = true
  end

  # Ma57 icntl(9): Controls the max number of iterative refinement steps.
  workspace.M.control.icntl[9] = 10

  # Ma57 control.cntl(3): If the norm of the scaled residuals does not decrease by a factor of at least cntl(3), 
  # then the iterative refinement stops.
  workspace.M.control.cntl[3] = one(eltype(u)) # Perform iterative refinement
  try
    ma57_solve!(workspace.M, u, workspace.x, workspace.dx, workspace.work, 10)
  catch e
    !(e isa HSL.Ma57Exception) && rethrow(e)
  end
  if any(isnan, workspace.x) || workspace.M.info.info[1] < 0
    workspace.status = :failed
    return
  end
end

function solve_system!(
  workspace::PenaltyMA57Workspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2<:CompactBFGSK2}
  workspace.status = :success
  H = workspace.H
  B = workspace.H.B
  n, m = workspace.n, workspace.m
  p = min(B._insert - 1, B._mem)
  x1, x2, x3, y1, y2 = H.x1, H.x2, H.x3, H.y1, H.y2
  Z1, Z2 = H.Z1, H.Z2
  Uk = @view B.Uk[:, 1:p]
  Vk = @view B.Vk[:, 1:p]

  # Step 0: Write (#TODO: we can use easily use QRMumps instead of LDLFactorization here...)
  # [B  Aᵀ] = [σI+ξI  Aᵀ] + [-U V]([U V])ᵀ
  # [A -αI] = [A     -αI] + [ 0 0]([0 0])
  # Hence,
  # [B  Aᵀ] = [σI+ξI  Aᵀ] + EFᵀ
  # [A -αI] = [A     -αI] + EFᵀ

  # Step 1: Factorize
  # [σI+ξI  Aᵀ]
  # [A     -αI]

  # Ma57 icntl(7): Controls the pivotting strategy.
  # Ma57 icntl(7): A value of 3 performs no pivoting: for a sufficiently large value of σ and a positive value of α,
  # The matrix is strongly factorizable, so we don't need pivoting.
  workspace.M.control.icntl[7] = 3

  if !workspace.factorized
    try
      ma57_factorize!(workspace.M)
    catch e
      !(e isa HSL.Ma57Exception) && rethrow(e)
    end
    workspace._n_fact += 1
  end

  # Ma57 info(1): a negative value is an error in the factorization.
  # Ma57 info(1): a value of 4 indicates that the matrix is singular.
  if workspace.M.info.info[1] < 0 || workspace.M.info.info[1] == 4
    workspace.status = :failed
    return
  else
    workspace.factorized = true
  end

  # Step 2: Compute
  # [x₁] = [σI+ξI  Aᵀ]⁻¹[u]
  # [x₁] = [A     -αI]  [u]

  # Ma57 icntl(9): Controls the max number of iterative refinement steps.
  workspace.M.control.icntl[9] = 10

  # Ma57 control.cntl(3): If the norm of the scaled residuals does not decrease by a factor of at least cntl(3), 
  # then the iterative refinement stops.
  workspace.M.control.cntl[3] = one(eltype(u)) # Perform iterative refinement
  try
    ma57_solve!(workspace.M, u, x1, workspace.dx, workspace.work, 10)
  catch e
    !(e isa HSL.Ma57Exception) && rethrow(e)
  end
  if any(isnan, x1) || workspace.M.info.info[1] < 0
    workspace.status = :failed
    return
  end

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
  try
    ma57_solve!(workspace.M, Z1, workspace._qn_work)
  catch e
    !(e isa HSL.Ma57Exception) && rethrow(e)
  end
  if any(isnan, Z1) || workspace.M.info.info[1] < 0
    workspace.status = :failed
    return
  end

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
  try
    ma57_solve!(workspace.M, x2, x3, workspace.dx, workspace.work, 10)
  catch e
    !(e isa HSL.Ma57Exception) && rethrow(e)
  end
  if any(isnan, x3) || workspace.M.info.info[1] < 0
    workspace.status = :failed
    return
  end

  # Step 8:
  # [B  Aᵀ]⁻¹[u] = x₁ - x₃ 
  # [A -αI]  [u] = x₁ - x₃
  workspace.x .= x1 .- x3
end

function get_solution!(
  x::V,
  workspace::PenaltyMA57Workspace{WP,K2},
) where {V<:AbstractVector,WP,K2}
  x .= workspace.x
end

function get_status(workspace::PenaltyMA57Workspace{WP,K2}) where {WP,K2}
  return workspace.status
end

function get_inertia(workspace::PenaltyMA57Workspace{WP,K2}) where {WP,K2}

  n, m = workspace.n, workspace.m
  (npos, nzero, nneg) = (0, 0, 0)

  nneg = workspace.M.info.info[24]
  rank = workspace.M.info.info[25]
  nzero = n + m - rank
  npos = n + m - nzero - nneg

  return npos, nzero, nneg
end
