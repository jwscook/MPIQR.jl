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

"""
    LinearAlgebra.mul!(C::SubArray, A::Adjoint, B::SubArray)

Required for the factorization

Faster than and functionally identical to this:
```julia
CC = CuArray(C)
CUDA.CUBLAS.gemm!('C', 'N', true, A', CuArray(B), false, CC)
C .= CC
```

'Arguments'
- C::SubArray
- A::Adjoint
- B::SubArray
"""
function LinearAlgebra.mul!(
    C::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
    A::Adjoint{T, CuArray{T, 1, CUDA.DeviceMemory}},
    B::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, CuArray{Int64, 1, CUDA.DeviceMemory}}, false}
    ) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}}

  Hj = parent(A)   # SubArray{T,1,<:CuArray}

  @assert size(C,1) == 1
  @assert size(B,1) == length(Hj)
  @assert size(C,2) == size(B,2)

  n = size(B,2)

  threads = 256
  blocks  = cld(n, threads)

  @cuda threads=threads blocks=blocks _rowvec_mat_mul_kernel!(C, Hj, B, n)

  return C
end
function _rowvec_mat_mul_kernel!(C, Hj, H, n)
  j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  if j <= n
    acc = zero(eltype(C))
    @inbounds for i = 1:n
      acc += conj(Hj[i]) * H[i, j]
    end
    C[1, j] = acc
  end
  return
end

function MPIQR.normandscale!(
    v::CuArray{T,1,CUDA.DeviceMemory},
    α::CuArray{T,1,CUDA.DeviceMemory},
    j::Int64) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64}}

  n = length(v)
  threads = 256
  blocks = cld(n, threads)

  partial = CuArray{real(T)}(undef, blocks)

  shmem = threads * sizeof(real(T))

  # 1) blockwise norm^2
  @cuda threads=threads blocks=blocks shmem=shmem norm2_blocks!(v, partial, n)

  # 2) final reduction + scale
  @cuda threads=threads blocks=1 shmem=shmem finalize_normandscale!(v, α, j, partial, n)

  return nothing
end

function norm2_blocks!(v, partial, n)
  T = eltype(v)

  tid = threadIdx().x
  idx = (blockIdx().x - 1) * blockDim().x + tid

  shared = @cuDynamicSharedMem(real(T), blockDim().x)

  acc = zero(real(T))
  idx <= n && (acc = abs2(v[idx]))
  shared[tid] = acc
  sync_threads()

  stride = blockDim().x >>> 1
  while stride > 0
    tid <= stride && (shared[tid] += shared[tid + stride])
    sync_threads()
    stride >>>= 1
  end

  tid == 1 && (partial[blockIdx().x] = shared[1])
  return nothing
end

function finalize_normandscale!(v, α, j, partial, n)
  T = eltype(v)

  tid = threadIdx().x
  shared = @cuDynamicSharedMem(real(T), blockDim().x)

  # reduce partial sums
  acc = zero(real(T))
  i = tid
  while i <= length(partial)
    acc += partial[i]
    i += blockDim().x
  end
  shared[tid] = acc
  sync_threads()

  stride = blockDim().x >>> 1
  while stride > 0
    tid <= stride && (shared[tid] += shared[tid + stride])
    sync_threads()
    stride >>>= 1
  end

  if tid == 1
    s = sqrt(shared[1])
    v1 = v[1]
    αval = s * MPIQR.alphafactor(v1)
    α[j] = αval
    v[1] = v1 - αval

    f = inv(sqrt(s * (s + abs(v1))))
    shared[1] = f
  end
  sync_threads()

  f = shared[1]

  # scale vector
  idx = tid
  while idx <= n
    v[idx] *= f
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
    H::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
    Hj::CuArray{T, 1, CUDA.DeviceMemory},
    z::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
    α::Number, β::Number) where T<:Union{Float32, Float64, ComplexF32, ComplexF64}
  Hc = CuArray(H)
  mul!(Hc, Hj, CuArray(z), T(α), T(β))
  copyto!(H, Hc)
  return H
end

end #module


