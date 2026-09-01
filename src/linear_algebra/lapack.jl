# Bunch-Kaufman (symmetric indefinite) factorization and solve, via direct
# ccalls into LAPACK's [sd]sytrf/[sd]sytrs. Used to factor small, dense
# matrices (e.g. a MUMPS Schur complement) without going through
# LinearAlgebra.bunchkaufman!, which allocates more than we need here.
for (sytrf, sytrs, T) in
  ((:ssytrf_, :ssytrs_, :Float32), (:dsytrf_, :dsytrs_, :Float64))
  @eval begin

    # Factorizes the symmetric matrix `a` (only the `uplo` triangle is
    # referenced/overwritten) in place as a = U*D*U' (uplo='U') or
    # a = L*D*L' (uplo='L'), where D is block diagonal with 1x1 and 2x2
    # blocks. `ipiv` (length n) encodes the block/pivot structure and is
    # required both to solve with `sytrs!` and to recover the inertia of
    # `a` from the diagonal blocks stored in `a` afterwards (see
    # `bunchkaufman_inertia`).
    function sytrf!(
      uplo::Char,
      n::BlasInt,
      a::AbstractMatrix{$T},
      lda::BlasInt,
      ipiv::AbstractVector{BlasInt},
      work::AbstractVector{$T},
      lwork::BlasInt,
      info::Base.RefValue{BlasInt},
    )
      ccall((@blasfunc($sytrf), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt},
             Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong),
            uplo, n, a, lda, ipiv, work, lwork, info, 1)
      return info[]
    end

    # Solves a*x = b in place (result stored in `b`), given the
    # factorization of `a` previously computed by `sytrf!` (same `uplo`
    # and `ipiv`).
    function sytrs!(
      uplo::Char,
      n::BlasInt,
      nrhs::BlasInt,
      a::AbstractMatrix{$T},
      lda::BlasInt,
      ipiv::AbstractVector{BlasInt},
      b::AbstractVecOrMat{$T},
      ldb::BlasInt,
      info::Base.RefValue{BlasInt},
    )
      ccall((@blasfunc($sytrs), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt},
             Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong),
            uplo, n, nrhs, a, lda, ipiv, b, ldb, info, 1)
      return info[]
    end
  end
end

"""
    bunchkaufman_inertia(a, ipiv, n; uplo = 'U') -> (npos, nzero, nneg)

Recover the inertia of the `n×n` symmetric matrix previously factored in
place into `a` by `sytrf!(uplo, ...)`, from the block-diagonal `D` factor
(stored on the diagonal of `a`) and the pivot structure `ipiv`.

Walks the 1x1/2x2 diagonal blocks of `D`, from `k = n` down to `k = 1` for
`uplo = 'U'` (the convention `sytrf!` uses to number blocks in that case):
each 1x1 block contributes its own sign; each 2x2 block `[d11 d21; d21 d22]`
contributes two eigenvalues of opposite sign if `det < 0`, two of the sign
of the trace if `det > 0`, and one zero eigenvalue (plus one of the sign of
the trace) if `det == 0`.
"""
function bunchkaufman_inertia(
  a::AbstractMatrix{T},
  ipiv::AbstractVector{BlasInt},
  n::Integer;
  uplo::Char = 'U',
) where {T}
  uplo == 'U' || throw(ArgumentError("bunchkaufman_inertia currently only supports uplo = 'U'"))

  npos, nzero, nneg = 0, 0, 0
  k = n
  while k >= 1
    if ipiv[k] > 0
      # 1x1 pivot block: D(k,k) = a[k,k].
      d = a[k, k]
      if d > 0
        npos += 1
      elseif d < 0
        nneg += 1
      else
        nzero += 1
      end
      k -= 1
    else
      # 2x2 pivot block spanning (k-1,k): ipiv[k] == ipiv[k-1] < 0.
      d11 = a[k-1, k-1]
      d22 = a[k, k]
      d21 = a[k-1, k]
      det = d11 * d22 - d21^2
      tr = d11 + d22
      if det < 0
        npos += 1
        nneg += 1
      elseif det > 0
        if tr > 0
          npos += 2
        else
          nneg += 2
        end
      else
        # Singular 2x2 block: one zero eigenvalue, the other sign(tr).
        nzero += 1
        if tr > 0
          npos += 1
        elseif tr < 0
          nneg += 1
        else
          nzero += 1
        end
      end
      k -= 2
    end
  end
  return npos, nzero, nneg
end
