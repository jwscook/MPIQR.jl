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

  v_parent = parent(v)
  row_offset = first(v.indices[1]) - 1
  col_offset = first(v.indices[2]) - 1
  nrows_parent = size(v_parent, 1)
  base_offset = col_offset * nrows_parent + row_offset

  # Optimized thread/block configuration
  threads = min(256, nextpow(2, m))
  blocks = cld(m, threads * 4)  # Process 4 elements per thread

  partial = CuArray{Float64}(undef, blocks)
  shmem = threads * sizeof(Float64)

  @cuda threads=threads blocks=blocks shmem=shmem norm2_blocks_subarray!(
    v_parent, partial, m, base_offset
  )

  @cuda threads=threads blocks=1 shmem=shmem finalize_normandscale_subarray!(
    v_parent, α, j, partial, m, base_offset
  )

  return nothing
end


function norm2_blocks_subarray!(v_parent, partial, m, offset)
  tid = threadIdx().x
  idx = (blockIdx().x - 1) * blockDim().x * 4 + tid
  shared = @cuDynamicSharedMem(Float64, blockDim().x)

  # Process 4 elements per thread (loop unrolling)
  acc = zero(real(eltype(v_parent)))
  for i in 0:3
    @inbounds if idx + i * blockDim().x <= m
      acc += abs2(v_parent[offset + idx + i * blockDim().x])
    end
  end

  shared[tid] = acc
  sync_threads()

  # Optimized reduction: unroll last warp
  for i in (512, 256, 128)
    if blockDim().x >= i
      tid <= i ÷ 2 && (shared[tid] += shared[tid + i ÷ 2])
      sync_threads()
    end
  end

  # Final warp reduction (no sync needed)
  if tid <= 32
    blockDim().x >= 64 && (shared[tid] += shared[tid + 32])
    for i in (16, 8, 4, 2)
      tid <= i && (shared[tid] += shared[tid + i])
    end
    tid == 1 && (partial[blockIdx().x] = shared[1] + shared[2])
  end

  return nothing
end

function finalize_normandscale_subarray!(v_parent, α, j, partial, m, offset)
  tid = threadIdx().x
  shared = @cuDynamicSharedMem(Float64, blockDim().x)

  # Grid-stride loop for partial reduction
  acc = zero(eltype(partial))
  i = tid
  len = length(partial)
  @inbounds while i <= len
    acc += partial[i]
    i += blockDim().x
  end
  shared[tid] = acc
  sync_threads()

  # Optimized reduction with unrolled warp
  for i in (512, 256, 128)
    if blockDim().x >= i
      tid <= i ÷ 2 && (shared[tid] += shared[tid + i ÷ 2])
      sync_threads()
    end
  end

  if tid <= 32
    blockDim().x >= 64 && (shared[tid] += shared[tid + 32])
    for i in (16, 8, 4, 2)
      tid <= i && (shared[tid] += shared[tid + i])
    end
    if tid == 1
      s = sqrt(shared[1] + shared[2])
      @inbounds v1 = v_parent[offset + 1]
      αval = s * MPIQR.alphafactor(v1)
      α[j] = αval
      @inbounds v_parent[offset + 1] = v1 - αval
      shared[1] = inv(sqrt(s * (s + abs(v1))))
    end
  end
  sync_threads()

  f = shared[1]

  # Grid-stride loop with vectorization
  idx = tid
  @inbounds while idx <= m
    v_parent[offset + idx] *= f
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

function LinearAlgebra.mul!(
    y::CuArray{T, 2, CUDA.DeviceMemory},
    Hj_adj::Adjoint{T, SubArray{T, 2, CuArray{ComplexF64, 2, CUDA.DeviceMemory}, <:Tuple, false}},
    H::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}
    ) where T<:Union{Float32, Float64, ComplexF32, ComplexF64}
  CUDA.CUBLAS.gemm!('C', 'N', one(T), Hj_adj', H, zero(T), y) # y = Hj' * H
  return y
end

function LinearAlgebra.mul!(
    H::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
    Hj::CuArray{T, 1, CUDA.DeviceMemory},
    z::CuArray{T, 2, CUDA.DeviceMemory},
    α::Int, β::Bool) where T<:Union{ComplexF32, ComplexF64}

  @assert α == -1 && β == 1 "This kernel is optimized for α=-1, β=1 only"
  @assert size(z, 1) == 1 "z must have first dimension = 1"

  # Reshape z to a vector (view, no copy)
  z_vec = view(z, 1, :)

  # Use CUBLAS ger!/geru! for rank-1 update: H = H + α * Hj * z_vec'
  # For real types: use ger!
  # For complex types: use geru! (unconjugated) or gerc! (conjugated)
  CUDA.CUBLAS.geru!(T(-1), Hj, z_vec, H)

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

function LinearAlgebra.mul!(
    H::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
    Hj::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, Base.Slice{Base.OneTo{Int64}}}, false},
    z::CuArray{T, 2, CUDA.DeviceMemory},
    α::Int, β::Bool) where T<:Union{Float32, Float64, ComplexF32, ComplexF64}

  @assert size(H, 1) == size(Hj, 1)
  @assert size(H, 2) == size(z, 2)
  @assert size(Hj, 2) == size(z, 1)

  # Use CUBLAS gemm! for: C = α * op(A) * op(B) + β * C
  # We want: H = -1 * Hj * z + 1 * H (i.e., H -= Hj * z)
  # gemm! signature: gemm!(tA, tB, alpha, A, B, beta, C)
  # where tA, tB are 'N' (no transpose), 'T' (transpose), or 'C' (conjugate transpose)
  CUDA.CUBLAS.gemm!('N', 'N', T(α), Hj, z, T(β), H)

  return H
end

end #module

