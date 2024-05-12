module MPIQR

using LinearAlgebra, Base.Threads, Base.Iterators
using Distributed, MPI, MPIClusterManagers
using Octavian

alphafactor(x::Real) = -sign(x)
alphafactor(x::Complex) = -exp(im * angle(x))

struct MPIQRMatrix{T} <: AbstractMatrix{T}
  localmatrix::Matrix{T}
  globalsize::Tuple{Int64, Int64}
  localcolumns::Vector{Int}
  columnlookup::Vector{Int}
  colsets::Vector{Set{Int}}
  blocksize::Int
  rank::Int64
  comm::MPI.Comm
  commsize::Int64
end

function localcolumns(rnk, n, blocksize, commsize)
  return vcat(collect(partition(collect(1:n), blocksize))[rnk + 1:commsize:end]...)
end
function MPIQRMatrix(localmatrix::AbstractMatrix, globalsize; blocksize=1, comm = MPI.COMM_WORLD)
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
  return MPIQRMatrix(localmatrix, globalsize, localcols, columnlookup, colsets, blocksize, rnk, comm, commsize)
end
columnowner(A::MPIQRMatrix, j) = findfirst(in(j, s) for s in A.colsets) - 1

Base.size(A::MPIQRMatrix) = A.globalsize
Base.getindex(A::MPIQRMatrix, i, j) = A.localmatrix[i, localcolindex(A, j)]

function Base.setindex!(A::MPIQRMatrix, v::Number, i, j)
  return A.localmatrix[i, localcolindex(A, j)] = v
end

localsize(A::MPIQRMatrix, dim=nothing) = size(A.localmatrix, dim)
localcolindex(A::MPIQRMatrix, j) = A.columnlookup[j]
localcolsize(A::MPIQRMatrix, j) = length(localcolindex(A, j))
blocksize(A::MPIQRMatrix) = A.blocksize

const IsBitsUnion = Union{Float32, Float64, ComplexF32, ComplexF64,
  Vector{Float32}, Vector{Float64}, Vector{ComplexF32}, Vector{ComplexF64}}

function hotloop!(H::MPIQRMatrix, Hj::AbstractVector, y, j, ja, jz, m, n)
  ja > n && return nothing
  iters = intersect(H.localcolumns, ja:jz)
  @inbounds @views @threads :dynamic for jj in iters
    s = sum(i->conj(Hj[i]) * H[i, jj], j:m)
    H[j:m, jj] .-= Hj[j:m] * s
  end
  return nothing
end
function hotloop!(H::MPIQRMatrix{T}, Hj::AbstractVector, y, j, ja, jz, m, n) where {T<:IsBitsUnion}
  jz > n && return nothing
  js = intersect(H.localcolumns, ja:jz)
  isempty(js) && return nothing
  lja = localcolindex(H, js[1])
  ljz = localcolindex(H, js[end])
  ljs = lja:ljz
  ll = length(ljs)
  mul!(view(y, 1:ll), view(H.localmatrix, j:m, lja:ljz)', view(Hj, j:m))
  # ger!(alpha, x, y, A) A = alpha*x*y' + A.
  BLAS.ger!(-one(T), view(Hj, j:m), view(y, 1:ll), view(H.localmatrix, j:m, lja:ljz))
  return nothing
end
function hotloop!(H::MPIQRMatrix, Hj::AbstractMatrix, y, j, ja, jz, m, n)
  for Δk in 0:size(Hj, 2)-1
    hotloop!(H, view(Hj, :, 1 + Δk), y, j + Δk, ja, jz, m, n)
  end
end

function householder!(H::AbstractMatrix{T}) where T
  m, n = size(H)
  α = zeros(T, min(m, n)) # Diagonal of R
  bs = blocksize(H)
  Hj = zeros(T, m, bs) # one column so that it works with Octavian
  t1 = t2 = t3 = t4 = t5 = 0.0
  y = zeros(eltype(H), localcolsize(H, 3:n))

  j = 1
  src = columnowner(H, j)
  if H.rank == src
    @inbounds @views copyto!(Hj[j:m, :], H[j:m, j:j - 1 + bs])
  end
  MPI.Bcast!(Hj, H.comm; root=src)

  tmp = zeros(T, m * bs)
  @inbounds @views for j in 1:bs:n

    sleep(H.rank * 0.1)
    for i in eachindex(Hj)
        @show j, H.rank, i, Hj[i]
    end

    colowner = columnowner(H, j)
    for Δj in 0:bs-1
      t1 += @elapsed @views begin
        s = norm(Hj[j + Δj:m, 1 + Δj])
        α[j + Δj] = s * alphafactor(Hj[j + Δj, 1 + Δj])
        f = 1 / sqrt(s * (s + abs(Hj[j + Δj, 1 + Δj])))
        Hj[j + Δj, 1 + Δj] -= α[j + Δj]
        Hj[j + Δj:m, 1 + Δj] .*= f
      end
      t2 += @elapsed if H.rank == colowner
        @views copyto!(H[j + Δj:m, j + Δj], Hj[j + Δj:m, 1 + Δj])
      end
      t3 += @elapsed hotloop!(H, Hj, y, j + Δj, j + 1 + Δj, j - 1 + 2bs, m, n)
    end
    @show "A", H.rank, size(tmp)

    t4 += @elapsed if j + bs <= n
      resize!(tmp, (m - (j - 1 + bs)) * bs)
    @show "B", H.rank, size(tmp)
      src = columnowner(H, j + bs)
      reqs = Vector{MPI.Request}()
      if H.rank == src
        for (cj, jj) in enumerate(j + bs:j - 1 + 2bs), (ci, ii) in enumerate(j+bs:m)
          tmp[ci, cj] = H[ii, jj]
          t = tmp[ci, cj]
          @show j, src, ci, cj, t
        end
        for r in filter(!=(src), 0:H.commsize-1)
          @show src, size(tmp), (j + bs) + n * r
          push!(reqs, MPI.Isend(tmp, H.comm; dest=r, tag=(j + bs) + n * r))
        end
      else
        @show H.rank, size(tmp),(j + bs) + n * H.rank
        push!(reqs, MPI.Irecv!(tmp, H.comm; source=src, tag=(j + bs) + n * H.rank))
      end
    end

    t3 += @elapsed hotloop!(H, Hj, y, j, j + 2bs, n, m, n)

    @show "C", H.rank, size(tmp)
    t5 += @elapsed if j + bs <= n
      MPI.Waitall(reqs)
      sleep(H.rank * 0.1)
      for cj in 1:bs, (ci, ii) in enumerate(j+bs:m)
        Hj[ii, cj] = tmp[ci, cj]
        t = tmp[ci, cj]#Hj[ii, cj]
        @show j, H.rank, cj, ii, t
      end
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
    tc += @elapsed b1 .= MPI.Allreduce(b1, +, H.comm)
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
    tf += @elapsed bi = MPI.Allreduce(bi, +, H.comm)
    b[i] -= bi
    b[i] /= α[i]
  end
  ts = (ta, tb, tc, td, te, tf)
  sts = sum(ts)
  H.rank == 0 && @show (ts ./ sum(ts)..., sum(ts))
  return b[1:n]
end

end

