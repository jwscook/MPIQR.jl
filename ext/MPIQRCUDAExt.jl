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
  AA = transpose(CuArray{T, 2, CUDA.DeviceMemory}(transpose(A))) # transpose back, allocate, re-transpose
  return mul!(C, AA, B)
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
  return C .= A * CuArray(B)
end

"""
    LinearAlgebra.mul!(C::SubArray, A::CuArray, B::SubArray)

Needed for the factorization

Faster than and functionally identical to this:
```julia
CC = CuArray(C)
CUDA.CUBLAS.gemm!('N', 'N', α, A, CuArray(B), β, CC)
C .= CC
```

'Arguments'
- C::SubArray
- A::CuArray
- B::SubArray
"""
function LinearAlgebra.mul!(
    C::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, CuArray{Int64, 1, CUDA.DeviceMemory}}, false},
    A::CuArray{T, 1, CUDA.DeviceMemory},
    B::SubArray{T, 2, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false},
    α::Number=1, β::Number=0) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}}
  return C .= A * B .* α .+ β .* C
end

#
#function LinearAlgebra.axpy!(α::CuArray{T, 1, CUDA.DeviceMemory},
#    x::SubArray{T, 1, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, Int64}, true},
#    y::SubArray{T, 1, CuArray{T, 2, CUDA.DeviceMemory}, Tuple{UnitRange{Int64}, Int64}, true}
#    ) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}}
#  return y .+= α .* x
#end
#function LinearAlgebra.scal!(a, x::CuArray)
#  return mul!(x, x, a)
#end
end #module
