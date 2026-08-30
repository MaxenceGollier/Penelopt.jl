mutable struct PenaltyLacpackWorkspace{T <: Real} <: PenaltyDirectWorkspace
  m::BlasInt                 # rows of A
  n::BlasInt                 # cols of A (= rows/cols since we assume square)
  a::Matrix{T}                # the factorized matrix, overwritten in-place by getrf!
  lda::BlasInt                # leading dimension of a
  ipiv::Vector{BlasInt}        # pivot indices, length min(m,n)
  info::Base.RefValue{BlasInt} # LAPACK status output
  nrhs::BlasInt                # number of right-hand sides for getrs!
  ldb::BlasInt                 # leading dimension of b
end

function PenaltyLacpackWorkspace(A::Matrix{T}; nrhs::Integer = 1) where {T <: Real}
  m, n = size(A)
  m == n || throw(ArgumentError("getrs! requires a square matrix, got $m x $n"))
  lda  = max(1, m)
  ipiv = Vector{BlasInt}(undef, m)
  info = Ref{BlasInt}(0)
  ldb  = max(1, n)
  return PenaltyLacpackWorkspace{T}(m, n, A, lda, ipiv, info, BlasInt(nrhs), BlasInt(ldb))
end

for (getrf, getrs, T) in
  ((:sgetrf_, :sgetrs_, :Float32), (:dgetrf_, :dgetrs_, :Float64))
  @eval begin

    # getrf
    function getrf!(
      workspace::PenaltyLacpackWorkspace{ST},
    ) where{ST <: $T}
      return ccall((@blasfunc($getrf), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
                      workspace.m, workspace.n, workspace.a, workspace.lda, workspace.ipiv, workspace.info)
    end

    # getrs
    function getrs!(
      trans::Char,
      workspace::PenaltyLacpackWorkspace{ST},
      b::AbstractVector{ST},
    ) where{ST <: $T}
      return ccall((@blasfunc($getrs), libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt},
                Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong),
                trans, workspace.n, nrhs, workspace.a, workspace.lda, workspace.ipiv, b, workspace.ldb, workspace.info, 1)
    end
  end
end