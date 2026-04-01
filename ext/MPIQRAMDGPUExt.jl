module MPIQRAMDGPUExt
using LinearAlgebra
using MPIQR
using AMDGPU

const HIPBuffer = AMDGPU.Runtime.Mem.HIPBuffer

MPIQR.maybeview(A::ROCArray, args...) = A[args...]

function Base.parent(M::Type{SubArray{T, N, ROCArray{T, N, HIPBuffer}, Tuple{U, U}, false}}
    ) where {T<:Number, N, U<:UnitRange{<:Integer}}
  return M.parameters[3]
end

# ---------------------------------------------------------------------------
# mul! for transposed subarray times subarray vector
# ---------------------------------------------------------------------------
function LinearAlgebra.mul!(
    bi::ROCArray{T, 1, HIPBuffer},
    A::Transpose{T, SubArray{T, 2, ROCArray{T, 2, HIPBuffer}, Tuple{SubArray{Int64, 1, ROCArray{Int64, 1, HIPBuffer}, Tuple{UnitRange{Int64}}, true}, Base.Slice{Base.OneTo{Int64}}}, false}},
    x::SubArray{T, 1, ROCArray{T, 2, HIPBuffer}, Tuple{Int64, UnitRange{Int64}}, true}
    ) where T <: Union{Float32, Float64, ComplexF32, ComplexF64}
  A_cont = similar(ROCArray{T}, size(parent(A)))
  A_cont .= parent(A)
  x_cont = similar(ROCArray{T}, size(x))
  x_cont .= x
  AMDGPU.rocBLAS.gemv!('T', one(T), A_cont, x_cont, zero(T), bi)
  return bi
end

function kernel_mul_T_sub!(C, A, Arowinds, Acolinds, B, Brow, Bcolinds, n)
  i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
  i > length(C) && return nothing
  acc = zero(eltype(C))
  @inbounds for k = 1:n
    a = A[Arowinds[k], Acolinds[i]]
    b = B[Brow, Bcolinds[k]]
    acc += a * b
  end
  @inbounds C[i] = acc
  return nothing
end

# ---------------------------------------------------------------------------
# normandscale!
# ---------------------------------------------------------------------------
function MPIQR.normandscale!(
    v::SubArray{T, 2, ROCArray{T, 2, HIPBuffer}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
    α::ROCArray{T, 1, HIPBuffer},
    j::Int64) where T<:Union{Float32, Float64, ComplexF32, ComplexF64}

  m = size(v, 1)
  @assert size(v, 2) == 1 "v must be a column vector (n=1)"

  v_parent   = parent(v)
  row_offset = first(v.indices[1]) - 1
  col_offset = first(v.indices[2]) - 1
  nrows_parent = size(v_parent, 1)
  base_offset  = col_offset * nrows_parent + row_offset

  groupsize = min(256, nextpow(2, m))
  gridsize  = cld(m, groupsize * 4)
  shmem = sizeof(T) * groupsize

  partial = ROCArray{Float64}(undef, gridsize)

  # Two-pass: partial norms, then finalize + scale
  @roc groupsize=groupsize gridsize=gridsize shmem=shmem norm2_gridsize_subarray!(
    v_parent, partial, m, base_offset
  )

  @roc groupsize=groupsize gridsize=1 shmem=shmem finalize_normandscale_subarray!(
    v_parent, α, j, partial, m, base_offset
  )

  return nothing
end

@inline function reduce_sum_shared!(shared, tid)
  for i in (512, 256, 128, 64, 32, 16, 8, 4, 2)
    if workgroupDim().x >= i
      tid <= i ÷ 2 && (shared[tid] += shared[tid + i ÷ 2])
    end
    sync_workgroup()
  end
end

function norm2_gridsize_subarray!(v_parent, partial, m, offset)
  tid = workitemIdx().x
  idx = (workgroupIdx().x - 1) * workgroupDim().x * 4 + tid
  shared = @ROCDynamicLocalArray(Float64, workgroupDim().x)
  acc = 0.0
  for i in 0:3
    @inbounds if idx + i * workgroupDim().x <= m
      acc += abs2(v_parent[offset + idx + i * workgroupDim().x])
    end
  end
  shared[tid] = acc
  sync_workgroup()
  reduce_sum_shared!(shared, tid)
  tid == 1 && (partial[workgroupIdx().x] = shared[1])
  sync_workgroup()
  return nothing
end

function finalize_normandscale_subarray!(v_parent, α, j, partial, m, offset)
  tid = workitemIdx().x
  shared = @ROCDynamicLocalArray(Float64, workgroupDim().x)
  acc = 0.0
  i = tid
  len = length(partial)
  @inbounds while i <= len
    acc += partial[i]
    i += workgroupDim().x
  end
  shared[tid] = acc
  sync_workgroup()
  reduce_sum_shared!(shared, tid)
  if tid == 1
    s = sqrt(shared[1])
    @inbounds v1 = v_parent[offset + 1]
    αval = s * MPIQR.alphafactor(v1)
    α[j] = αval
    @inbounds v_parent[offset + 1] = v1 - αval
    shared[1] = inv(sqrt(s * (s + abs(v1))))
  end
  sync_workgroup()
  f = shared[1]
  idx = tid
  @inbounds while idx <= m
    v_parent[offset + idx] *= f
    idx += workgroupDim().x
  end
  return nothing
end

# ---------------------------------------------------------------------------
# fill_diagonal_kernel!
# ---------------------------------------------------------------------------
function fill_diagonal_kernel!(A, n, value=one(eltype(A)))
  i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
  i <= n && (A[i, i] = value)
  return nothing
end

# ---------------------------------------------------------------------------
# ldiv! via rocBLAS trsm
# ---------------------------------------------------------------------------
function LinearAlgebra.ldiv!(
    A::ROCArray{T, 2, HIPBuffer},
    B::UnitLowerTriangular{T, ROCArray{T, 2, HIPBuffer}},
    _::Diagonal{Bool, Vector{Bool}}) where T<:Union{Float32, Float64, ComplexF32, ComplexF64}

  n = size(A, 1)
  @assert size(A) == size(B) == (n, n)

  groupsize = 256
  gridsize  = cld(n, groupsize)
  @roc groupsize=groupsize gridsize=gridsize fill_diagonal_kernel!(A, n)

  # Solve B * X = A  (unit lower triangular)
  AMDGPU.rocBLAS.trsm!('L', 'L', 'N', 'U', T(1), B.data, A)

  return A
end

# ---------------------------------------------------------------------------
# mul! adjoint subarray × subarray  (gemm)
# ---------------------------------------------------------------------------
function LinearAlgebra.mul!(
    y::ROCArray{T, 2, HIPBuffer},
    Hj_adj::Adjoint{T, SubArray{T, 2, ROCArray{ComplexF64, 2, HIPBuffer}, <:Tuple, false}},
    H::SubArray{T, 2, ROCArray{T, 2, HIPBuffer}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}
    ) where T<:Union{Float32, Float64, ComplexF32, ComplexF64}
  AMDGPU.rocBLAS.gemm!('C', 'N', one(T), Hj_adj', H, zero(T), y)
  return y
end

# ---------------------------------------------------------------------------
# mul! rank-1 update: H -= Hj * z'  (geru)
# ---------------------------------------------------------------------------
function LinearAlgebra.mul!(
    H::SubArray{T, 2, ROCArray{T, 2, HIPBuffer}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
    Hj::ROCArray{T, 1, HIPBuffer},
    z::ROCArray{T, 2, HIPBuffer},
    α::Int, β::Bool) where T<:Union{ComplexF32, ComplexF64}

  @assert α == -1 && β == 1
  @assert size(z, 1) == 1
  z_vec = view(z, 1, :)
  AMDGPU.rocBLAS.geru!(T(-1), Hj, z_vec, H)
  return H
end

# ---------------------------------------------------------------------------
# mul! subarray × ROCArray gemm (H -= Hj * z)
# ---------------------------------------------------------------------------
function LinearAlgebra.mul!(
    H::SubArray{T, 2, ROCArray{T, 2, HIPBuffer}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
    Hj::SubArray{T, 2, ROCArray{T, 2, HIPBuffer}, Tuple{UnitRange{Int64}, Base.Slice{Base.OneTo{Int64}}}, false},
    z::ROCArray{T, 2, HIPBuffer},
    α::Int, β::Bool) where T<:Union{Float32, Float64, ComplexF32, ComplexF64}

  @assert size(H, 1) == size(Hj, 1)
  @assert size(H, 2) == size(z, 2)
  @assert size(Hj, 2) == size(z, 1)

  AMDGPU.rocBLAS.gemm!('N', 'N', T(α), Hj, z, T(β), H)
  return H
end

# Adjoint 1D ROCArray  ×  2D SubArray  →  ROCArray row (as 1×n matrix)
function Base.:*(
    Hv::Adjoint{T, ROCArray{T, 1, HIPBuffer}},
    B::SubArray{T, 2, ROCArray{T, 2, HIPBuffer}, Tuple{UnitRange{Int64}, Base.Slice{Base.OneTo{Int64}}}, false}
    ) where T <: Union{Float32, Float64, ComplexF32, ComplexF64}
  k, n = size(B)
  z = similar(ROCArray{T}, (1, n))
  hv_mat = reshape(parent(Hv), k, 1)  # k×1 column
  B_cont = similar(ROCArray{T}, size(B))
  B_cont .= B
  # gemm!('C','N',...) conjugates hv_mat, giving (1×k)^C * (k×n) = (1×n)
  AMDGPU.rocBLAS.gemm!('C', 'N', one(T), hv_mat, B_cont, zero(T), z)
  return z
end

function LinearAlgebra.axpby!(
    a::T,
    x::ROCArray{T, 1, HIPBuffer},
    b::T,
    y::SubArray{T, 1, ROCArray{T, 2, HIPBuffer}, Tuple{Int64, Base.Slice{Base.OneTo{Int64}}}, true}
    ) where T <: Union{Float32, Float64, ComplexF32, ComplexF64}
  n = length(y)
  @assert length(x) == n
  @roc groupsize=256 gridsize=cld(n, 256) _axpby_kernel!(a, x, b, y, n)
  return y
end

function _axpby_kernel!(a, x, b, y, n)
  i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
  i > n && return nothing
  @inbounds y[i] = a * x[i] + b * y[i]
  return nothing
end

end # module
