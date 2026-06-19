# Compute a cheap upper bound on the smallest eigenvalue of H, which is used to set the primal inertia σ
# λmin(H) <= minᵢ eᵢᵀ H eᵢ = min_i H[i, i]
function lambda_min_upper_bound(H::SparseMatrixCOO{T}) where{T}
  rows, cols, vals = H.rows, H.cols, H.vals
  nnz = length(vals)
  upper_bound = T(Inf)
  @inbounds for i = 1:nnz
    if rows[i] == cols[i]
      upper_bound = min(upper_bound, vals[i])
    end
  end
  return upper_bound
end

function lambda_min_upper_bound(H::CompactBFGS{T}) where{T}
  return T(Inf)
end