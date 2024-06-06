module MPIQR

using LinearAlgebra, Base.Threads, Base.Iterators
using Distributed, MPI, MPIClusterManagers, Hwloc
using ProgressMeter

struct FakeProgress end
ProgressMeter.next!(::FakeProgress; showvalues=nothing) = nothing

const L2CACHESIZEBYTES = Hwloc.cachesize().L2

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
localcolumns(A::MPIQRMatrix) = A.localcolumns
localmatrix(A::MPIQRMatrix) = A.localmatrix

function MPIQRMatrix(localmatrix::AbstractMatrix, globalsize; blocksize=1, comm = MPI.COMM_WORLD)
  @assert blocksize >= 1
  rnk = MPI.Comm_rank(comm)
  commsize = MPI.Comm_size(comm)
  @assert commsize >= 1
  m, n = globalsize
  @assert mod(n, blocksize) == 0
  localcols = localcolumns(rnk, n, blocksize, commsize)
  @assert length(localcols) > 0
  @assert minimum(localcols) >= 1
  @assert maximum(localcols) <= n
  @assert issorted(localcols)
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
function Base.:*(A::MPIQRMatrix{T}, x::AbstractVector{U}) where {T,U}
  y = A.localmatrix * x[A.localcolumns]
  return MPI.Allreduce(y, +, A.comm)
end

localsize(A::MPIQRMatrix, dim=nothing) = size(A.localmatrix, dim)
localcolindex(A::MPIQRMatrix, j) = A.columnlookup[j]
localcolsize(A::MPIQRMatrix, j) = length(localcolindex(A, j))
blocksize(A::MPIQRMatrix) = A.blocksize
struct ColumnIntersectionIterator
  localcolumns::Vector{Int}
  indices::UnitRange{Int}
end
Base.@propagate_inbounds function Base.iterate(
    iter::ColumnIntersectionIterator, state=0)
  isempty(iter.indices) && return nothing
  if state >= length(iter.indices)
    return nothing
  else
    state += 1
    return (iter.localcolumns[iter.indices[state]], state)
  end
end
Base.first(cii::ColumnIntersectionIterator) = cii.localcolumns[first(cii.indices)]
Base.last(cii::ColumnIntersectionIterator) = cii.localcolumns[last(cii.indices)]
Base.length(cii::ColumnIntersectionIterator) = length(cii.indices)

function Base.intersect(A::MPIQRMatrix, cols)
  indexa = searchsortedfirst(A.localcolumns, first(cols))
  indexz = searchsortedlast(A.localcolumns, last(cols))
  indexa = indexa > length(A.localcolumns) ? length(A.localcolumns) + 1 : indexa
  indexz = indexz > length(A.localcolumns) ? 0 : indexz
  indices = indexa:indexz
  output = ColumnIntersectionIterator(A.localcolumns, indices)
  return output
end

function threadedcopyto!(a, b)
  @assert length(a) == length(b)
#  if Threads.nthreads() == 1 || length(a) < 2^12
    @inbounds @simd for i in eachindex(a, b)
      a[i] = b[i]
    end
#  else
#    @inbounds @threads :static for i in eachindex(a, b)
#      a[i] = b[i]
#    end
#  end
end

const IsBitsUnion = Union{Float32, Float64, ComplexF32, ComplexF64,
  Vector{Float32}, Vector{Float64}, Vector{ComplexF32}, Vector{ComplexF64}}

function hotloop!(H::AbstractMatrix, Hj::AbstractVector, y)
  @inbounds @views @threads :dynamic for jj in 1:size(H, 2)
    s = sum(i->conj(Hj[i]) * H[i, jj], eachindex(Hj))
    H[:, jj] .-= Hj * s
  end
  return nothing
end

function hotloop!(H::AbstractMatrix{T}, Hj::AbstractVector, y) where {T<:IsBitsUnion}
  isempty(y) && return nothing
#  mul!(y, H', Hj) # same as BLAS.gemv!('C', true, H, Hj, false, y)
#  BLAS.ger!(-one(T), Hj, y, H) # ger!(alpha, x, y, A) A = alpha*x*y' + A.
  ntile = clamp(sizeof(H) ÷ L2CACHESIZEBYTES, 1, size(H, 2))
  for j in Base.Iterators.partition(1:size(H, 2), ntile)
    mul!(view(y, j), view(H, :, j)', Hj)
    # ger!(alpha, x, y, A) A = alpha*x*y' + A.
    BLAS.ger!(-one(T), Hj, view(y, j), view(H, :, j))
  end
  return nothing
end
function hotloopviews(H::MPIQRMatrix, Hj::AbstractVector, y, j, ja, jz, m, n,
    js = intersect(H, ja:jz))
  lja = localcolindex(H, first(js))
  ljz = localcolindex(H, last(js))
  ll = length(lja:ljz)
  return (view(H.localmatrix, j:m, lja:ljz), view(Hj, j:m), view(y, 1:ll))
end

function hotloop!(H::MPIQRMatrix, Hj::AbstractVector, y, j, ja, jz, m, n)
  js = intersect(H, ja:jz)
  isempty(js) && return nothing
  viewH, viewHj, viewy = hotloopviews(H, Hj, y, j, ja, jz, m, n, js)
  hotloop!(viewH, viewHj, viewy)
  return nothing
end

function hotloop!(H::MPIQRMatrix, Hj::AbstractMatrix, y, j, ja, jz, m, n)
  bs = blocksize(H)
  for Δk in 0:size(Hj, 2)-1
    hotloop!(H, view(Hj, :, 1 + Δk), y, j + Δk, ja, jz, m, n)
  end
end

function householder!(H::MPIQRMatrix{T}, α=zeros(T, size(H, 2)); verbose=false,
    progress=FakeProgress()) where T
  m, n = size(H)
  @assert m > n
  bs = blocksize(H) # the blocksize / tilesize of contiguous columns on each rank
  Hj = zeros(T, m, bs) # the H column(s)
  Hjcopy = bs > 1 ? zeros(T, m) : Hj # copy of the H column(s)
  t1 = t2 = t3 = t4 = t5 = 0.0
  # work array for the BLAS call
  y = zeros(eltype(H), localcolsize(H, 1:n))

  # send the first column(s) of H to Hj on all ranks
  j = 1
  src = columnowner(H, j)
  if H.rank == src
    @views threadedcopyto!(Hj[j:m, :], H[j:m, j:j - 1 + bs])
  end
  MPI.Bcast!(Hj, H.comm; root=src)

  tmp = zeros(T, m * bs)
  @inbounds @views for j in 1:bs:n
    colowner = columnowner(H, j)

    # process all the first bs column(s) of H
    @inbounds for Δj in 0:bs-1
      t1 += @elapsed @views begin
        s = norm(Hj[j + Δj:m, 1 + Δj])
        α[j + Δj] = s * alphafactor(Hj[j + Δj, 1 + Δj])
        f = 1 / sqrt(s * (s + abs(Hj[j + Δj, 1 + Δj])))
        Hj[j + Δj, 1 + Δj] -= α[j + Δj]
        Hj[j + Δj:m, 1 + Δj] .*= f
      end

      t2 += @elapsed bs > 1 && threadedcopyto!(view(Hjcopy, j+Δj:m, 1), view(Hj, j+Δj:m, 1 + Δj)) # prevent data race
      t3 += @elapsed hotloop!(view(Hj, j+Δj:m, 1 + Δj:bs), view(Hjcopy, j+Δj:m, 1), view(y, 1 + Δj:bs))

      t2 += @elapsed if H.rank == colowner
        @views threadedcopyto!(H[j + Δj:m, j + Δj:j-1+bs], Hj[j + Δj:m, 1 + Δj:bs])
      end
    end

    # now next do the next column to make it ready for the next iteration of the loop
    t3 += @elapsed hotloop!(H, Hj, y, j, j + bs, j - 1 + 2bs, m, n)

    # if it's not the last iteration send the next iterations Hj to all ranks
    t4 += @elapsed if j + bs <= n
      resize!(tmp, (m - (j - 1 + bs)) * bs)
      src = columnowner(H, j + bs)
      reqs = Vector{MPI.Request}()
      if H.rank == src
        k = 0
        for (cj, jj) in enumerate(j + bs:j - 1 + 2bs), (ci, ii) in enumerate(j+bs:m)
          @inbounds tmp[k+=1] = H[ii, jj]
        end
        for r in filter(!=(src), 0:H.commsize-1)
          push!(reqs, MPI.Isend(tmp, H.comm; dest=r, tag=(j + bs) + n * r))
        end
      else
        push!(reqs, MPI.Irecv!(tmp, H.comm; source=src, tag=(j + bs) + n * H.rank))
      end
    end

    # Apply Hj to all the columns of H to the right of this column + 2 blocks
    t3 += @elapsed hotloop!(H, Hj, y, j, j + 2bs, n, m, n)

    # Now receive next iterations Hj
    t5 += @elapsed if j + bs <= n
      MPI.Waitall(reqs)
      viewHj = view(Hj, j+bs:m, 1:bs)
      linearviewHj = reshape(viewHj, length(tmp))
      threadedcopyto!(linearviewHj, tmp)
    end
    iszero(H.rank) && next!(progress)
  end
  ts = (t1, t2, t3, t4, t5)
  verbose && H.rank == 0 && @show (ts ./ sum(ts)..., sum(ts))
  return MPIQRStruct(H, α)
end


function solve_householder!(b, H, α; verbose=false)
  m, n = size(H)
  # multuply by Q' ...
  b1 = zeros(eltype(b), length(b))
  ta = tb = tc = td = te = tf = 0.0
  @inbounds @views for j in 1:n
    if in(j, H.localcolumns)
      ta += @elapsed s = dot(H[j:m, j], b[j:m])
      tb += @elapsed b1[j:m] .= H[j:m, j] .* s
    else
      b1[j:m] .= 0
    end
    tc += @elapsed MPI.Allreduce!(b1, +, H.comm)
    b[j:m] .-= b1[j:m]
    b1[j] = 0
  end
  # now that b holds the value of Q'b
  # we may back sub with R
  td += @elapsed MPI.Barrier(H.comm)
  @inbounds @views for i in n:-1:1
    #bis = zeros(eltype(b), Threads.nthreads())
    #te += @elapsed @threads for j in collect(intersect(H, i+1:n))
    #  bis[Threads.threadid()] += H[i, j] * b[j]
    #end
    #bi = sum(bis)
    bi = zero(eltype(b))
    te += @elapsed @inbounds for j in intersect(H, i+1:n)
      bi += H[i, j] * b[j]
    end
    tf += @elapsed bi = MPI.Allreduce(bi, +, H.comm)
    b[i] -= bi
    b[i] /= α[i]
  end
  ts = (ta, tb, tc, td, te, tf)
  verbose && H.rank == 0 && @show (ts ./ sum(ts)..., sum(ts))
  return b[1:n]
end

struct MPIQRStruct{T1, T2}
  A::T1
  α::T2
end

MPIQRStruct(A::MPIQRMatrix) = MPIQRStruct(A, zeros(eltype(A), size(A, 2)))

function LinearAlgebra.qr!(A::MPIQRMatrix; progress=FakeProgress())
  H = MPIQRStruct(A)
  householder!(H.A, H.α; progress=progress)
  return H
end

LinearAlgebra.:(\)(H::MPIQRStruct, b) = solve_householder!(b, H.A, H.α)

end

