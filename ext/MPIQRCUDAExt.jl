module MPIQRCUDAExt
using LinearAlgebra
using MPIQR
using CUDA

MPIQR.maybeview(A::CuArray, args...) = A[args...]

function Base.parent(M::Type{SubArray{T, N, CuArray{T, N, CUDA.DeviceMemory}, Tuple{U, U}, false}}
    ) where {T<:Number, N, U<:UnitRange{<:Integer}}
  return M.parameters[3]
end

"""
    LinearAlgebra.mul!(C::CuArray, A::Transpose, B::CuArray)

Required for the solve

'Arguments'
- C::CuArray
- A::Transpose
- B::CuArray
"""
function LinearAlgebra.mul!(
    C::CuArray{T, 1, CUDA.DeviceMemory},
    A::Transpose{T, SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{SubArray{Int64, 1, CuArray{Int64, 1, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}}, true}, Base.Slice{Base.OneTo{Int64}}}, false}},
    B::SubArray{T, 1, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{Int64, UnitRange{Int64}}, true}
    ) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}}

  # underlying parent and indexing metadata
  Aview = parent(A)              # SubArray
  Aparent = parent(Aview)        # CuArray
  Ainds = Aview.indices          # (row_inds, col_inds)

  Bparent = parent(B)
  Binds = B.indices              # (row, col_range)

  m = length(C)
  n = length(B)

  @cuda threads=256 blocks=cld(m,256) kernel_mul_T_sub!(
    C,
    Aparent, Ainds[1], Ainds[2],
    Bparent, Binds[1], Binds[2],
    n
  )

  return C
end

function kernel_mul_T_sub!(C, A, Arowinds, Acolinds, B, Brow, Bcolinds, n)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  i > length(C) && return nothing
  acc = zero(eltype(C))
  @inbounds for k = 1:n
    # A is transposed: Aᵀ[i,k] = A[k,i]
    a = A[Arowinds[k], Acolinds[i]]
    b = B[Brow, Bcolinds[k]]
    acc += a * b
  end
  @inbounds C[i] = acc
  return nothing
end

function MPIQR.normandscale!(
    v::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
    α::CuArray{T, 1, CUDA.DeviceMemory},
    j::Int64) where T<:Union{Float32, Float64, ComplexF32, ComplexF64}

  m = size(v, 1)
  n = size(v, 2)
  @assert n == 1 "v must be a column vector (n=1)"

  # Get parent array and offsets
  v_parent = parent(v)
  row_offset = first(v.indices[1]) - 1
  col_offset = first(v.indices[2]) - 1
  nrows_parent = size(v_parent, 1)

  # Linear offset for the column in parent array
  base_offset = (col_offset) * nrows_parent + row_offset

  threads = 256
  blocks = cld(m, threads)
  partial = CuArray{Float64}(undef, blocks)
  shmem = threads * sizeof(Float64)

  # 1) blockwise norm^2
  @cuda threads=threads blocks=blocks shmem=shmem norm2_blocks_subarray!(
    v_parent, partial, m, base_offset
  )

  # 2) final reduction + scale
  @cuda threads=threads blocks=1 shmem=shmem finalize_normandscale_subarray!(
    v_parent, α, j, partial, m, base_offset
  )

  return nothing
end

function norm2_blocks_subarray!(v_parent, partial, m, offset)
  tid = threadIdx().x
  idx = (blockIdx().x - 1) * blockDim().x + tid
  shared = @cuDynamicSharedMem(Float64, blockDim().x)

  acc = 0.0
  if idx <= m
    @inbounds acc = abs2(v_parent[offset + idx])
  end

  shared[tid] = acc
  sync_threads()

  # Reduction in shared memory
  stride = blockDim().x >>> 1
  while stride > 0
    if tid <= stride
      shared[tid] += shared[tid + stride]
    end
    sync_threads()
    stride >>>= 1
  end

  if tid == 1
    partial[blockIdx().x] = shared[1]
  end

  return nothing
end

function finalize_normandscale_subarray!(v_parent, α, j, partial, m, offset)
  tid = threadIdx().x
  shared = @cuDynamicSharedMem(Float64, blockDim().x)

  # Reduce partial sums
  acc = 0.0
  i = tid
  while i <= length(partial)
    @inbounds acc += partial[i]
    i += blockDim().x
  end

  shared[tid] = acc
  sync_threads()

  # Final reduction
  stride = blockDim().x >>> 1
  while stride > 0
    if tid <= stride
      shared[tid] += shared[tid + stride]
    end
    sync_threads()
    stride >>>= 1
  end

  # Compute scaling factor and update first element
  if tid == 1
    s = sqrt(shared[1])
    @inbounds v1 = v_parent[offset + 1]
    αval = s * MPIQR.alphafactor(v1)
    α[j] = αval
    @inbounds v_parent[offset + 1] = v1 - αval
    f = inv(sqrt(s * (s + abs(v1))))
    shared[1] = f
  end

  sync_threads()
  f = shared[1]

  # Scale vector
  idx = tid
  while idx <= m
    @inbounds v_parent[offset + idx] *= f
    idx += blockDim().x
  end

  return nothing
end


function fill_diagonal_kernel!(A, n, value=one(eltype(A)))
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  i <= n && (A[i, i] = value)
  return nothing
end

# Interface to ldiv! that calls CUBLAS
function LinearAlgebra.ldiv!(
  A::CuArray{T, 2, CUDA.DeviceMemory},
  B::UnitLowerTriangular{T, CuArray{T, 2, CUDA.DeviceMemory}},
  _::Diagonal{Bool, Vector{Bool}}) where T<:Union{Float32, Float64, ComplexF32, ComplexF64}

  n = size(A, 1)
  @assert size(A) == size(B) == (n, n) "$(size(A)) != $(size(B)) !== $((n, n))"

  # A comes in as zeros, so we only need to set the diagonal to 1
  threads = 256
  blocks = cld(n, threads)
  @cuda threads=threads blocks=blocks fill_diagonal_kernel!(A, n)

  # Solve B * A = D (where D is the diagonal we just set)
  # using CUBLAS trsm: B * X = A, so X overwrites A
  # 'L' = left side, 'L' = lower triangular, 'N' = no transpose, 'U' = unit diagonal
  CUDA.CUBLAS.trsm!('L', 'L', 'N', 'U', T(1), B.data, A)

  return A
end

function LinearAlgebra.mul!( # fast
    y::CuArray{T, 2, CUDA.DeviceMemory},
    aHj::Adjoint{T, CuArray{T, 1, CUDA.DeviceMemory}},
    H::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}
    ) where T<:Union{Float32, Float64, ComplexF32, ComplexF64}


  Hj = parent(aHj)
  m = length(Hj)
  n = size(H, 2)

  @assert size(y, 1) == 1
  @assert size(y, 2) == n
  @assert size(H, 1) == m

  # Get SubArray parent and offsets
  H_parent = parent(H)
  H_row_offset = first(H.indices[1]) - 1
  H_col_offset = first(H.indices[2]) - 1

  threads = 256
  blocks = cld(n, threads)

  @cuda threads=threads blocks=blocks _rowvec_matmul_kernel!(
    y, Hj, H_parent, m, n, H_row_offset, H_col_offset
  )

  return y
end

function _rowvec_matmul_kernel!(y, Hj, H_parent, m, n, row_off, col_off)
  T = eltype(y)
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x

  if j <= n
    acc = zero(T)

    # Unrolled loop for better performance
    i = 1
    @inbounds while i <= m - 3
      h1 = conj(Hj[i])
      h2 = conj(Hj[i+1])
      h3 = conj(Hj[i+2])
      h4 = conj(Hj[i+3])

      acc += h1 * H_parent[row_off + i, col_off + j]
      acc += h2 * H_parent[row_off + i + 1, col_off + j]
      acc += h3 * H_parent[row_off + i + 2, col_off + j]
      acc += h4 * H_parent[row_off + i + 3, col_off + j]

      i += 4
    end

    @inbounds while i <= m
      acc += conj(Hj[i]) * H_parent[row_off + i, col_off + j]
      i += 1
    end

    y[1, j] = acc
  end

  return
end

function LinearAlgebra.mul!(
    H::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
    Hj::CuArray{T, 1, CUDA.DeviceMemory},
    z::CuArray{T, 2, CUDA.DeviceMemory},
    α::Int, β::Bool) where T<:Union{Float32, Float64, ComplexF32, ComplexF64}

  @assert α == -1 && β == 1 "This kernel is optimized for α=-1, β=1 only"
  @assert size(z, 1) == 1 "z must have first dimension = 1"

  # Reshape z to a vector (view, no copy)
  z_vec = view(z, 1, :)

  # Use CUBLAS ger!/geru! for rank-1 update: H = H + α * Hj * z_vec'
  # For real types: use ger!
  # For complex types: use geru! (unconjugated) or gerc! (conjugated)
  if T <: Union{Float32, Float64}
    CUDA.CUBLAS.ger!(T(-1), Hj, z_vec, H)
  else  # Complex types
    CUDA.CUBLAS.geru!(T(-1), Hj, z_vec, H)
  end

  return H
end

function Base.copyto!(
    a::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
    b::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}
    ) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}}

  @assert size(a, 2) == 1
  @assert size(b, 2) == 1
  @assert size(a, 1) == size(b, 1)
  @assert a.indices[1] == b.indices[1]

  m = size(a, 1)

  # For very small copies, just use the default
  if m < 1000
    a .= b
    return a
  end

  a_parent = parent(a)
  b_parent = parent(b)

  row_offset = first(a.indices[1]) - 1
  a_col = first(a.indices[2])
  b_col = first(b.indices[2])

  nrows_a = size(a_parent, 1)
  nrows_b = size(b_parent, 1)

  a_start = (a_col - 1) * nrows_a + row_offset + 1
  b_start = (b_col - 1) * nrows_b + row_offset + 1

  # Use CUDA's optimized unsafe_copyto! when possible
  if nrows_a == nrows_b && a_col == b_col && a_parent === b_parent
    # Same location, nothing to do
    return a
  end

  # Direct memory copy using GPU memcpy
  unsafe_copyto!(a_parent, a_start, b_parent, b_start, m)

  return a
end

#function Base.copyto!(
#    a::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
#    b::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}
#    ) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}}
#
#  @assert size(a, 2) == 1
#  @assert size(b, 2) == 1
#  @assert size(a, 1) == size(b, 1)
#  @assert a.indices[1] == b.indices[1] # a limitation of this
#
#  m = size(a, 1)
#
#  a_parent = parent(a)
#  b_parent = parent(b)
#
#  row_offset = first(a.indices[1]) - 1
#  a_col_offset = first(a.indices[2]) - 1
#  b_col_offset = first(b.indices[2]) - 1
#
#  nrows_a = size(a_parent, 1)
#  nrows_b = size(b_parent, 1)
#
#  a_base_offset = a_col_offset * nrows_a + row_offset
#  b_base_offset = b_col_offset * nrows_b + row_offset
#
#  threads = 256
#  blocks  = cld(m, threads)
#
#  @cuda threads=threads blocks=blocks fast_copy_kernel!(
#      a_parent, b_parent, m, a_base_offset, b_base_offset
#  )
#  return a
#end
#
#function fast_copy_kernel!(a, b, len, a_offset, b_offset)
#    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#    if i <= len
#        @inbounds a[a_offset + i] = b[b_offset + i]
#    end
#    return nothing
#end

function LinearAlgebra.mul!(
    H::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
    Hj::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, Base.Slice{Base.OneTo{Int64}}}, false},
    z::CuArray{T, 2, CUDA.DeviceMemory},
    α::Int, β::Bool) where T<:Union{Float32, Float64, ComplexF32, ComplexF64}

  @assert α == -1 && β == 1 "This kernel is optimized for α=-1, β=1 only"

  m = size(Hj, 1)
  k = size(Hj, 2)
  n = size(z, 2)

  @assert size(H, 1) == m
  @assert size(H, 2) == n
  @assert size(z, 1) == k

  H_parent = parent(H)
  Hj_parent = parent(Hj)

  H_row_offset = first(H.indices[1]) - 1
  H_col_offset = first(H.indices[2]) - 1
  Hj_row_offset = first(Hj.indices[1]) - 1

  threads = (16, 16)
  blocks = (cld(m, 16), cld(n, 16))

  @cuda threads=threads blocks=blocks _matmul_subtract_kernel!(
    H_parent, Hj_parent, z, m, k, n, H_row_offset, H_col_offset, Hj_row_offset
  )

  return H
end

function _matmul_subtract_kernel!(H_parent, Hj_parent, z, m, k, n,
                                  H_row_off, H_col_off, Hj_row_off)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

  if i <= m && j <= n
    acc = zero(eltype(H_parent))

    # 4x unrolled loop for better ILP
    l = 1
    @inbounds while l <= k - 7
      acc += Hj_parent[Hj_row_off + i, l]     * z[l, j]
      acc += Hj_parent[Hj_row_off + i, l + 1] * z[l + 1, j]
      acc += Hj_parent[Hj_row_off + i, l + 2] * z[l + 2, j]
      acc += Hj_parent[Hj_row_off + i, l + 3] * z[l + 3, j]
      acc += Hj_parent[Hj_row_off + i, l + 4] * z[l + 4, j]
      acc += Hj_parent[Hj_row_off + i, l + 5] * z[l + 5, j]
      acc += Hj_parent[Hj_row_off + i, l + 6] * z[l + 6, j]
      acc += Hj_parent[Hj_row_off + i, l + 7] * z[l + 7, j]
      l += 8
    end

    @inbounds while l <= k
      acc += Hj_parent[Hj_row_off + i, l] * z[l, j]
      l += 1
    end

    # H[i,j] -= acc (single FMA operation)
    @inbounds H_parent[H_row_off + i, H_col_off + j] -= acc
  end

  return nothing
end

function LinearAlgebra.mul!(
    y::CuArray{T, 2, CUDA.DeviceMemory},
    Hj_adj::Adjoint{T, SubArray{T, 2, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, <:Tuple, false}},
    H::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}
    ) where T<:Union{Float32, Float64, ComplexF32, ComplexF64}
  CUDA.CUBLAS.gemm!('C', 'N', one(T), Hj_adj', H, zero(T), y) # y = Hj' * H
  return y
end

end #module

