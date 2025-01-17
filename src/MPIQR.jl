module MPIQR

using LinearAlgebra, Base.Threads, Base.Iterators, Combinatorics
using MPI, MPIClusterManagers, ProgressMeter

alphafactor(x::Real) = -sign(x)
alphafactor(x::Complex) = -exp(im * angle(x))

struct MPIQRMatrix{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
  localmatrix::M
  globalsize::Tuple{Int64, Int64}
  localcolumns::Vector{Int}
  columnlookup::Vector{Int}
  colsets::Vector{Set{Int}}
  blocksize::Int
  rank::Int64
  comm::MPI.Comm
  commsize::Int64
end

function validblocksizes(numcols::Integer, commsize::Integer)::Vector{Int}
  iszero(numcols ÷ commsize) || return [0]
  return findall(iszero(numcols % i) for i in 1:(numcols ÷ commsize))
end

function localcolumns(rnk, n, blocksize, commsize)
  output = Vector{Int}(vcat(collect(partition(collect(1:n), blocksize))[rnk + 1:commsize:end]...))
  @assert isempty(output) || minimum(output) >= 1
  @assert isempty(output) || maximum(output) <= n
  @assert issorted(output)
  return output
end
localcolumns(A::MPIQRMatrix) = A.localcolumns
localmatrix(A::MPIQRMatrix) = A.localmatrix

function MPIQRMatrix(localmatrix::AbstractMatrix, globalsize; blocksize=1, comm = MPI.COMM_WORLD)
  @assert blocksize >= 1
  rnk = MPI.Comm_rank(comm)
  commsize = MPI.Comm_size(comm)
  @assert commsize >= 1
  m, n = globalsize
  localcols = localcolumns(rnk, n, blocksize, commsize)
  colsets = Vector{Set{Int}}()
  for r in 0:commsize-1
    push!(colsets, Set(localcolumns(r, n, blocksize, commsize)))
  end
  if size(localmatrix, 2) != length(localcols)
    throw(ArgumentError(
      "This rank's matrix must have the right number of local columns"))
  end
  @assert size(localmatrix, 2) == length(localcols)

  lookupop(j) = (x = searchsortedfirst(localcols, j); isnothing(x) ? 0 : x)
  columnlookup = Vector{Int}([lookupop(j) for j in 1:n])
  @assert minimum(columnlookup) >= 0
  @assert maximum(columnlookup) <= n
  return MPIQRMatrix(localmatrix, globalsize, localcols, columnlookup, colsets, blocksize, rnk, comm, commsize)
end

"""
    columnowner(A::MPIQRMatrix,j)::Int

Return the rank of the owner of column j of matrix A.

...
# Arguments
- `A::MPIQRMatrix`:
- `j::Int`:
...
"""
function columnowner(A::MPIQRMatrix, j)::Int
  for (i, cols) in enumerate(A.colsets)
    in(j, cols) && return i - 1
  end
  @assert false "Shouldn't be able to get here"
  return -1
end

Base.size(A::MPIQRMatrix) = A.globalsize
Base.size(A::MPIQRMatrix, i) = A.globalsize[i]
Base.getindex(A::MPIQRMatrix, i, j) = A.localmatrix[i, localcolindex(A, j)]

function Base.setindex!(A::MPIQRMatrix, v, i, j)
  return A.localmatrix[i, localcolindex(A, j)] = v
end

# define these for dispatch purposes
Base.:*(A::MPIQRMatrix, x::AbstractVector) = _mul(A, x)
Base.:*(A::MPIQRMatrix, x::AbstractMatrix) = _mul(A, x)
function _mul(A::MPIQR.MPIQRMatrix, x)
  y = A.localmatrix * view(x, A.localcolumns, :)
  return MPI.Allreduce(y, +, A.comm)
end

localsize(A::MPIQRMatrix, dim=nothing) = size(A.localmatrix, dim)
localcolindex(A::MPIQRMatrix, j) = A.columnlookup[j]
localcolsize(A::MPIQRMatrix, j) = length(localcolindex(A, j))
blocksize(A::MPIQRMatrix) = A.blocksize

struct FakeProgress end
ProgressMeter.next!(::FakeProgress; showvalues=nothing) = nothing
import ProgressMeter.Progress
function Progress(A::MPIQRMatrix, dt=1; kwargs...)
  return Progress(size(A, 2) ÷ A.blocksize, dt=dt; kwargs...)
end

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

function hotloopviews(H::MPIQRMatrix, Hj::AbstractMatrix, Hr, y, j, ja, jz, m, n,
    js = intersect(H, ja:jz))
  lja = localcolindex(H, first(js))
  ljz = localcolindex(H, last(js))
  ll = length(lja:ljz)
  return (view(H.localmatrix, j:m, lja:ljz), view(Hj, j:m, :), view(Hr, j:m, :), view(y, 1:ll, :))
end

function hotloop!(H::MPIQRMatrix, Hj::AbstractMatrix, Hr, y, j, ja, jz, m, n)
  js = intersect(H, ja:jz)
  isempty(js) && return nothing
  viewH, viewHj, viewHr, viewy = hotloopviews(H, Hj, Hr, y, j, ja, jz, m, n, js)
  hotloop!(viewH, viewHj, viewHr, viewy)
  return nothing
end

"""
    unrecursedcoeffs(N,A)

When one has

H(1) = H(0) - Hj(0) Hj(0)' H(0)
H(2) = H(1) - Hj(1) Hj(1)' H(1)
H(N) = H(N-1) - Hj(N-1) Hj(N-1)' H(N-1)

one can roll all of the multiplcations by Hj and Hj' into one matrix Hr
by multiplying and adding various combinations of the dot products of the
columns of Hj. This function calculates the combinations of these dot products.

...
# Arguments
- `N`:
- `A`:
...

# Example
```julia
```
"""
function unrecursedcoeffs(N, A)
  A >= N && return [[N, N]]
  output = [[A, N]]
  for i in 1:N-1, c in combinations(A+1:N-1, i)
    push!(output, [A, c..., N])
  end
  reverse!(output)
  return output
end

"""
    recurse!(H::AbstractMatrix,Hj::AbstractArray{T},Hr,y) where {T}

In stead of applying the columns of `Hj` to H` sequentially, it is better to
calculate the effective recursive action of `Hj` on `H` and store that in `Hr`
such that `Hr` can be applied to `H` in one big gemm call.

...
# Arguments
- `H::AbstractMatrix`: Apply the reflectors to this matrix
- `Hj::AbstractArray{T}`: The columns that could be applied to H albeit slowly.
- `Hr`: The effective recursed matrix of Hj to apply to H in one fast gemm call.
...

# Example
```julia
```
"""
function recurse!(H::AbstractMatrix, Hj::AbstractArray{T}, Hr, y) where {T}
  dots = zeros(T, size(Hj, 2), size(Hj, 2)) # faster than a dict
  # dots only really needs values in: for i in 1:size(Hj, 2), j in 1:i
  mul!(dots, Hj', Hj, true, false) # some of this calculation is redundent

  #BLAS.gemm!('C', 'N', true, H, Hj, false, y) # y = H' * Hj
  mul!(y, H', Hj, true, false) # y = H' * Hj

  copyto!(Hr, Hj)

  # this is complicated, I know, but the tests pass!
  # It's easier to verify by deploying this logic with symbolic quantities
  # and viewing the output
  @views @inbounds for ii in 1:size(Hj, 2)
     for i in ii + 1:size(Hj, 2)
      summand = zero(T)
      for urc in unrecursedcoeffs(i - 1, ii - 1)
        factor = -(-1)^length(urc) * one(T)
        @inbounds for j in 2:length(urc)
          factor *= dots[urc[j] + 1, urc[j-1] + 1]
        end
        summand += factor
      end
      axpy!(summand, view(Hj, :, i), view(Hr, :, ii))
    end
  end
end

function hotloop!(H::AbstractMatrix, Hj::AbstractArray{T}, Hr, y) where {T}

  recurse!(H, Hj, Hr, y)

  #BLAS.gemm!('N', 'C', -one(T), Hr, y, true, H) # H .-= Hj * y'
  mul!(H, Hr, y', -1, true) # H .-= Hj * y'

  return nothing
end


function householder!(H::MPIQRMatrix{T}, α=zeros(T, size(H, 2)); verbose=false,
    progress=FakeProgress()) where T
  m, n = size(H)
  @assert m >= n
  bs = blocksize(H) # the blocksize / tilesize of contiguous columns on each rank
  Hj = zeros(T, m, bs) # the H column(s)
  Hr = zeros(T, m, bs) # the H column(s)
  t1 = t2 = t3 = t4 = t5 = 0.0
  # work array for the BLAS call
  y = zeros(eltype(H), localcolsize(H, 1:n), bs)

  # send the first column(s) of H to Hj on all ranks
  j = 1
  src = columnowner(H, j)
  if H.rank == src
    @views copyto!(Hj[j:m, :], H[j:m, j:j - 1 + bs])
  end
  MPI.Bcast!(view(Hj, j:m, :), H.comm; root=src)

  tmp = zeros(T, m * bs)
  @inbounds @views for j in 1:bs:n
    colowner = columnowner(H, j)
    bs = min(bs, n - j + 1)
    bz = min(bs, n - j - bs + 1)

    # process all the first bs column(s) of H
    @inbounds for Δj in 0:bs-1
      t1 += @elapsed @views begin
        s = norm(Hj[j + Δj:m, 1 + Δj])
        α[j + Δj] = s * alphafactor(Hj[j + Δj, 1 + Δj])
        f = 1 / sqrt(s * (s + abs(Hj[j + Δj, 1 + Δj])))
        Hj[j:j + Δj - 1, 1 + Δj] .= 0
        Hj[j + Δj, 1 + Δj] -= α[j + Δj]
        Hj[j + Δj:m, 1 + Δj] .*= f
      end

      t3 += @elapsed hotloop!(view(Hj, j+Δj:m, 1 + Δj:bs), view(Hj, j+Δj:m, 1+Δj), view(Hr, j+Δj:m, 1), view(y, 1 + Δj:bs))

      t2 += @elapsed if H.rank == colowner
        @views copyto!(H[j + Δj:m, j + Δj:j-1+bs], Hj[j + Δj:m, 1 + Δj:bs])
      end
    end

    # now do the next blocksize of colums to ready it for the next iteration
    t3 += @elapsed hotloop!(H, Hj, Hr, y, j, j + bs, j - 1 + bs + bz, m, n)

    # if it's not the last iteration send the next iterations Hj to all ranks
    t4 += @elapsed if j + bs <= n
      # make tmp the right linear length so that it accommodate all the new Hj
      resize!(tmp, (m - (j - 1 + bs)) * bz)
      src = columnowner(H, j + bs)
      reqs = Vector{MPI.Request}()
      if H.rank == src
        @inbounds tmp .= reshape(view(H, j+bs:m, j+bs:j-1+bs+bz), length(tmp))
        for r in filter(!=(src), 0:H.commsize-1)
          push!(reqs, MPI.Isend(tmp, H.comm; dest=r, tag=j + bs))
        end
      else
        push!(reqs, MPI.Irecv!(tmp, H.comm; source=src, tag=j + bs))
      end
    end

    # Apply Hj to all the columns of H to the right of this column + 2 blocks
    t3 += @elapsed hotloop!(H, Hj, Hr, y, j, j + bs + bz, n, m, n)

    # Now receive next iterations Hj
    t5 += @elapsed if j + bs <= n
      MPI.Waitall(reqs)
      viewHj = view(Hj, j+bs:m, 1:bz)
      linearviewHj = reshape(viewHj, length(tmp))
      copyto!(linearviewHj, tmp) # mutates Hj
    end
    next!(progress)
  end
  ts = (t1, t2, t3, t4, t5)
  verbose && @show ts
  return MPIQRStruct(H, α)
end


function solve_householder!(b, H, α; progress=FakeProgress(), verbose=false)
  m, n = size(H)
  bs = blocksize(H)
  # multuply by Q' ...
  b1 = zeros(eltype(b), size(b))
  s = zeros(promote_type(eltype(b), eltype(H)), (1, size(b, 2)))
  ta = tb = tc = td = te = 0.0
  @inbounds @views for j in 1:bs:n
    b1[j:m, :] .= 0
    blockrank = columnowner(H, j)
    bs = min(bs, n - j + 1)
    if H.rank == blockrank
      for jj in 0:bs-1
        @assert columnowner(H, j) == blockrank
        ta += @elapsed s .= H[j+jj:m, j+jj]' * b[j+jj:m, :]
        tb += @elapsed b[j+jj:m, :] .-= H[j+jj:m, j+jj] * s
        tb += @elapsed b1[j+jj:m, :] .+= H[j+jj:m, j+jj] * s
      end
    end
    tc += @elapsed MPI.Allreduce!(b1, +, H.comm)
    if H.rank != blockrank
      b[j:m, :] .-= b1[j:m, :]
    end
  end
  # now that b holds the value of Q'b
  # we may back sub with R
  @inbounds @views b[n, :] ./= α[n] # because iteration doesnt start at n
  bi = zeros(eltype(b), size(b, 2))
  @inbounds @views for i in n-1:-1:1
    bi .= 0
    td += @elapsed @inbounds for j in intersect(H, i+1:n)
      @. bi += H[i, j] * b[j, :]
    end
    te += @elapsed bi = MPI.Allreduce(bi, +, H.comm)
    @. b[i, :] = (b[i, :] - bi) / α[i]
    next!(progress)
  end
  ts = (ta, tb, tc, td, te)
  verbose && @show ts
  return b[1:n, :]
end

struct MPIQRStruct{T1, T2}
  A::T1
  α::T2
end

MPIQRStruct(A::MPIQRMatrix) = MPIQRStruct(A, zeros(eltype(A), size(A, 2)))
Base.size(s::MPIQRStruct, i=nothing) = size(s.A, i)
Base.setindex!(s::MPIQRStruct, v, i, j) = setindex!(s.A, v, i, j)
Base.getindex(s::MPIQRStruct, i, j) = getindex(s.A, i, j)

Progress(s::MPIQRStruct, dt=1; kwargs...) = Progress(s.A, dt=dt; kwargs...)
localcolumns(s::MPIQRStruct) = localcolumns(s.A)

function LinearAlgebra.qr!(s::MPIQRStruct; progress=FakeProgress(), verbose=false)
  return qr!(s.A; progress=progress, verbose=verbose)
end

function LinearAlgebra.qr!(A::MPIQRMatrix; progress=FakeProgress(), verbose=false)
  H = MPIQRStruct(A)
  householder!(H.A, H.α; progress=progress, verbose=verbose)
  return H
end

function LinearAlgebra.ldiv!(H::MPIQRStruct, b; progress=FakeProgress(),
                             verbose=false)
  return solve_householder!(b, H.A, H.α; progress=progress, verbose=verbose)
end

function LinearAlgebra.ldiv!(x, H::MPIQRStruct, b; progress=FakeProgress(),
                             verbose=false)
  c = deepcopy(b) # TODO: make this ...
  solve_householder!(c, H.A, H.α; progress=progress, verbose=verbose)
  x .= c[1:size(x, 1), :] # .. and there this better
  return x
end

LinearAlgebra.:(\)(H::MPIQRStruct, b) = ldiv!(H, deepcopy(b))

end

