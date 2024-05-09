module MPIQR

using LinearAlgebra, Base.Threads, Base.Iterators
using Distributed, MPI, MPIClusterManagers
using Octavian

alphafactor(x::Real) = -sign(x)
alphafactor(x) = -exp(im * angle(x))

struct MPIMatrix{T} <: AbstractMatrix{T}
  localmatrix::Matrix{T}
  globalsize::Tuple{Int64, Int64}
  localcolumns::Vector{Int}
  columnlookup::Vector{Int}
  colsets::Vector{Set{Int}}
  rank::Int64
  comm::MPI.Comm
  commsize::Int64
end

function localcolumns(rnk, n, blocksize, commsize)
  return vcat(collect(partition(collect(1:n), blocksize))[rnk + 1:commsize:end]...)
end
function MPIMatrix(localmatrix::AbstractMatrix, globalsize; blocksize=1, comm = MPI.COMM_WORLD)
  @assert blocksize >= 1
  rnk = MPI.Comm_rank(comm)
  commsize = MPI.Comm_size(comm)
  @assert commsize >= 1
  m, n = globalsize
  @assert mod(n, blocksize) == 0
  localcols = localcolumns(rnk, n, blocksize, commsize)
  @assert minimum(localcols) >= 1
  @assert maximum(localcols) <= n
  colsets = Vector{Set{Int}}()
  for r in 0:commsize-1
    push!(colsets, Set(localcolumns(r, n, blocksize, commsize)))
  end
  @assert size(localmatrix, 2) == length(localcols)

  lookupop(j) = (x = searchsortedfirst(localcols, j); isnothing(x) ? 0 : x)
  columnlookup = Vector{Int}([lookupop(j) for j in 1:n])
  @assert minimum(columnlookup) >= 0
  @assert maximum(columnlookup) <= n
  return MPIMatrix(localmatrix, globalsize, localcols, columnlookup, colsets, rnk, comm, commsize)
end
columnowner(A::MPIMatrix, j) = findfirst(in(j, s) for s in A.colsets) - 1

Base.size(A::MPIMatrix) = A.globalsize
localsize(A::MPIMatrix, dim=nothing) = size(A.localmatrix, dim)
Base.getindex(A::MPIMatrix, i, j) = A.localmatrix[i, localcolindex(A, j)]
localcolindex(A::MPIMatrix, j) = A.columnlookup[j]
function Base.setindex!(A::MPIMatrix, v::Number, i, j)
  return A.localmatrix[i, localcolindex(A, j)] = v
end
localcolsize(A::MPIMatrix, j) = length(localcolindex(A, j))

function hotloop!(H, Hj, y, j, ja, jz, m, n)
  iters = intersect(H.localcolumns, ja:jz)
  @inbounds @views @threads :dynamic for jj in iters
    s = sum(i->conj(Hj[i]) * H[i, jj], j:m)
    H[j:m, jj] .-= Hj[j:m] * s
  end
  return nothing
end
function hotloop!(H::MPIMatrix{T}, Hj, y, j, ja, jz, m, n
    ) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
  jz > n && return nothing
  js = intersect(H.localcolumns, ja:jz)
  isempty(js) && return nothing
  lja = localcolindex(H, js[1])
  ljz = localcolindex(H, js[end])
  ljs = lja:ljz
  ll = length(ljs)
  mul!(view(y, 1:ll), view(H.localmatrix, j:m, lja:ljz)', view(Hj, j:m))
#  if T <: Union{Float32, Float64, ComplexF32, ComplexF64}
    # ger!(alpha, x, y, A) A = alpha*x*y' + A.
    BLAS.ger!(-one(T), view(Hj, j:m), view(y, 1:ll), view(H.localmatrix, j:m, lja:ljz))
#  else
#    # matmul!(C, A, B[, α, β, max_threads]) C = α * A * B + β * C
#    matmul!(view(H.localmatrix, j:m, lja:ljz), view(Hj, j:m, :), view(y, 1:ll)', -1, 1)
#  end
  return nothing
end

function householder!(H::AbstractMatrix{T}) where T
  m, n = size(H)
  α = zeros(T, min(m, n)) # Diagonal of R
  Hj = zeros(T, m, 1) # one column so that it works with Octavian
  t1 = t2 = t3 = t4 = t5 = 0.0
  y = zeros(eltype(H), localcolsize(H, 3:n))

  j = 1
  src = columnowner(H, j)
  if H.rank == src
    @views Hj[j:m] .= H[j:m, j]
  end
  if isbits(T)
    MPI.Bcast!(Hj, H.comm; root=src)
  else
    Hj .= MPI.bcast(Hj, H.comm; root=src)
  end

  tmp = zeros(T, m)
  @inbounds @views for j in 1:n

    t1 += @elapsed @views begin
      s = norm(Hj[j:m])
      α[j] = s * alphafactor(Hj[j])
      f = 1 / sqrt(s * (s + abs(Hj[j])))
      Hj[j] -= α[j]
      Hj[j:m] .*= f
    end

    t2 += @elapsed if H.rank == columnowner(H, j)
      @views H[j:m, j] .= Hj[j:m]
    end

    t3 += @elapsed hotloop!(H, Hj, y, j, j + 1, j + 1, m, n)

    t4 += @elapsed if j + 1 <= n
      resize!(tmp, m - j)
      src = columnowner(H, j + 1)
      req = if H.rank == src
        tmp .= view(H, j+1:m, j + 1)
        for r in filter(!=(src), 0:H.commsize-1)
          MPI.Isend(tmp, H.comm; dest=r, tag=(j + 1) + n * r)
        end
        MPI.Request()
      else
        MPI.Irecv!(tmp, H.comm; source=src, tag=(j + 1) + n * H.rank)
      end
    end

    t3 += @elapsed hotloop!(H, Hj, y, j, j + 2, n, m, n)

    t5 += @elapsed if j + 1 <= n
      MPI.Wait(req)
      MPI.Barrier(H.comm)
      @views Hj[j+1:m] .= tmp
    end
  end
  ts = (t1, t2, t3, t4, t5)
  sts = sum(ts)
  H.rank == 0 && @show (ts ./ sum(ts)..., sum(ts))
  return (H, α)
end


# function householder!(H::AbstractMatrix{T}) where T
#   m, n = size(H)
#   α = zeros(T, min(m, n)) # Diagonal of R
#   Hj = zeros(T, m)
#   t1 = t2 = t3 = t4 = t5 = 0.0
#
#   j = 1
#   src = columnowner(H, j)
#   if H.rank == src
#     Hj[j:m] .= H[j:m, j]
#     for r in filter(!=(src), 0:H.commsize-1)
#       MPI.Isend(Hj[j:m], H.comm; dest=r, tag=j + n * r)
#     end
#   end
#
#   tmp = zeros(T, m)
#   t5 += @elapsed @inbounds @views for j in 1:n
#
#     src = columnowner(H, j)
#     resize!(tmp, m - j + 1)
#     t3 += @elapsed if H.rank != src
#         MPI.Recv!(tmp, H.comm; source=src, tag=j + n * H.rank)
#         @views Hj[j:m] .= tmp
#     else
#       @views Hj[j:m] .= H[j:m, j]
#     end
#
#     t1 += @elapsed @views begin
#       s = norm(Hj[j:m])
#       α[j] = s * alphafactor(Hj[j])
#       f = 1 / sqrt(s * (s + abs(Hj[j])))
#       Hj[j] -= α[j]
#       Hj[j:m] .*= f
#     end
#     t2 += @elapsed if H.rank == src
#       @views H[j:m, j] .= Hj[j:m]
#     end
#
#     t3 += @elapsed @views @threads :dynamic for jj in intersect(H.localcolumns, j+1:j+1)
#       s = sum(i->conj(Hj[i]) * H[i, jj], j:m)
#       @views @simd for i in j:m
#         H[i, jj] -= Hj[i] * s
#       end
#     end
#     t4 += @elapsed if j + 1 <= n
#       src = columnowner(H, j + 1)
#       if H.rank == src
#         resize!(tmp, m - j)
#         tmp .= view(H, j+1:m, j + 1)
#         for r in filter(!=(src), 0:H.commsize-1)
#           MPI.Isend(tmp, H.comm; dest=r, tag=(j + 1) + n * r)
#         end
#       end
#     end
#     t4 += @elapsed @views @threads :dynamic for jj in intersect(H.localcolumns, j+2:n)
#       #s = sum(i->conj(Hj[i]) * H[i, jj], j:m)
#       s = dot(view(Hj, j:m), view(H, j:m, jj))
#       H[j:m, jj] .-= Hj[j:m] * s
#     end
#
#   end
#   H.rank == 0 && @show t1, t2, t3, t4, t5
#
#   return (H, α)
# end
#
# function householder!(A::AbstractMatrix{T}) where T
#   m, n = size(A)
#   H = A
#   α = zeros(T, min(m, n)) # Diagonal of R
#   Hj = zeros(T, m)
#   t1 = t2 = t3 = t4 = t5 = 0.0
#   t5 += @elapsed @inbounds @views for j in 1:n
#     jinA = in(j, A.localcolumns)
#     t1 += @elapsed if jinA
#       s = norm(H[j:m, j])
#       α[j] = s * alphafactor(H[j, j])
#       f = 1 / sqrt(s * (s + abs(H[j, j])))
#       H[j, j] -= α[j]
#       H[j:m, j] .*= f
#     end
#     t2 += @elapsed jinArnk = columnowner(H, j)
#
#     t3 += @elapsed Hj[j:m] .= MPI.bcast(H[j:m, j], jinArnk, A.comm)
#
#     t4 += @elapsed @views @threads :dynamic for jj in intersect(A.localcolumns, j+1:n)
#       s = sum(i->conj(Hj[i]) * H[i, jj], j:m)
#       H[j:m, jj] .-= Hj[j:m] * s
#     end
#   end
#   t6 = @elapsed α = MPI.Allreduce(α, +, A.comm)
#   H.rank == 0 && @show t1, t2, t3, t4, t5, t6
#   return (H, α)
# end
function mpiallreduce(bi::T, op::F, H
    ) where {T <: Union{AbstractVector{<:Float64}, AbstractVector{Float64},
                        AbstractVector{<:ComplexF32}, AbstractVector{<:ComplexF64},
                        Float32, Float64, ComplexF32, ComplexF64}, F}
  bi = MPI.Allreduce(bi, op, H.comm)
  return bi
end
function mpiallreduce(bi::T, op::F, H) where {T, F}
  recvbuf = zero(bi)
  if 0 == H.rank
    for r in 1:H.commsize-1
      MPI.recv(recvbuf, H.comm; source=r, tag=r)
      bi += recvbuf
    end
  else
    MPI.send(bi, H.comm; dest=0, tag=0)
  end
  if 0 == H.rank
    for r in 1:H.commsize-1
      MPI.send(bi, H.comm; dest=r, tag=r)
    end
  else
    MPI.recv!(bi, H.comm; src=0, tag=0)
  end
  return bi
end

function solve_householder!(b, H, α)
  m, n = size(H)
  # multuply by Q' ...
  b1 = zeros(eltype(b), length(b))
  ta = tb = tc = td = te = tf = 0.0
  @inbounds @views for j in 1:n
    if in(j, H.localcolumns)
      ta += @elapsed s = dot(H[j:m, j], b[j:m])
      tb += @elapsed b1[j:m] .= H[j:m, j] .* s
    end
    tc += @elapsed b1 .= mpiallreduce(b1, +, H)#.comm)
    b[j:m] .-= b1[j:m]
    b1[j:m] .= 0
  end
  # now that b holds the value of Q'b
  # we may back sub with R
  td += @elapsed MPI.Barrier(H.comm)
  @inbounds @views for i in n:-1:1
    bi = zero(eltype(b))
    te += @elapsed for j in intersect(i+1:n, H.localcolumns)
      bi += H[i, j] * b[j]
    end
    tf += @elapsed bi = mpiallreduce(bi, +, H)#.comm)
    b[i] -= bi
    b[i] /= α[i]
  end
  ts = (ta, tb, tc, td, te, tf)
  sts = sum(ts)
  H.rank == 0 && @show (ts ./ sum(ts)..., sum(ts))
  return b[1:n]
end

end

