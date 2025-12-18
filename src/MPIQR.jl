module MPIQR

using LinearAlgebra, Base.Threads, Base.Iterators, Combinatorics
using MPI, MPIClusterManagers, ProgressMeter

alphafactor(x::Real) = -sign(x)
alphafactor(x::Complex) = -exp(im * angle(x))

struct MPIQRMatrix{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
  localmatrix::M
  globalsize::Tuple{Int64, Int64}
  localcolumns::Vector{Int} # list of global column indices on this localmatrix
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
  output = reduce(vcat, collect(partition(collect(1:n), blocksize))[rnk + 1:commsize:end]; init=Int[])
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

  lookupop(j) = (x = findfirst(isequal(j), localcols); isnothing(x) ? 0 : x)
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
Base.size(A::MPIQRMatrix, i::Integer) = A.globalsize[i]
Base.getindex(A::MPIQRMatrix, i, j) = A.localmatrix[i, localcolindex(A, j)]
Base.view(A::MPIQRMatrix, i, j) = view(A.localmatrix, i, localcolindex(A, j)) # helps out GPUs

function Base.setindex!(A::MPIQRMatrix, v, i, j)
  return A.localmatrix[i, localcolindex(A, j)] = v
end

# define these for dispatch purposes
Base.:*(A::MPIQRMatrix{T,M}, x::AbstractVector) where {T,M} = _mul(A, x)
Base.:*(A::MPIQRMatrix{T,M}, x::AbstractMatrix) where {T,M} = _mul(A, x)
LinearAlgebra.mul!(C, A::MPIQRMatrix, B) = _mul!(C, A, B)
maybeview(x, is, js) = view(x, is, js) # allow dispatch on type of x, lest x doesnt do views
function _mul(A::MPIQR.MPIQRMatrix, x)
  T = promote_type(eltype(A), eltype(x))
  return _mul!(similar(x, T, size(A, 1), size(x, 2)), A, x)
end
function _mul!(y, A::MPIQR.MPIQRMatrix, x)
  mul!(y, A.localmatrix, maybeview(x, A.localcolumns, :))
  return MPI.Allreduce!(y, +, A.comm)
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
    iter::ColumnIntersectionIterator, state=0)::Union{Nothing,Tuple{Int, Int}}
  isempty(iter.indices) && return nothing
  if state >= length(iter.indices)
    return nothing
  else
    state += 1
    return (iter.localcolumns[iter.indices[state]], state)
  end
end
Base.first(cii::ColumnIntersectionIterator)::Int = cii.localcolumns[first(cii.indices)]
Base.last(cii::ColumnIntersectionIterator)::Int = cii.localcolumns[last(cii.indices)]
Base.length(cii::ColumnIntersectionIterator) = length(cii.indices)

function Base.intersect(A::MPIQRMatrix, cols)
  indexa = searchsortedfirst(A.localcolumns, first(cols))
  indexz = searchsortedlast(A.localcolumns, last(cols))
  return ColumnIntersectionIterator(A.localcolumns, indexa:indexz)
end
function hotloopviews(H::MPIQRMatrix, Hj::AbstractMatrix, y, jm, js)
  ljaz = localcolindex(H, first(js)):localcolindex(H, last(js))
  ll = length(ljaz)
  return (view(H.localmatrix, jm, ljaz), view(Hj, jm, :), view(y, 1:ll, :))
end

function hotloop!(H::MPIQRMatrix, work, jm, jaz)
  js = intersect(H, jaz)
  isempty(js) && return nothing
  viewH, viewHj, viewy = hotloopviews(H, work.Hj, work.y, jm, js)
  hotloop!(viewH, (Hj=viewHj, y=viewy, dots=work.dots, coeffs=work.coeffs))
  return nothing
end

"""
    hotloop!(H::AbstractMatrix,work) where {T}

Does the main part of the decomposition algorithm. A lot of the logic is
to calculate the effective action of `Hj` on `H`, so
that the bulk of the work is done in fast level-3 BLAS calls. Note that
Hj is needed after this function call, so can't be overwritten.

When one has

H(1) = H(0) - Hj(0) Hj(0)' H(0)
H(2) = H(1) - Hj(1) Hj(1)' H(1)
H(N) = H(N-1) - Hj(N-1) Hj(N-1)' H(N-1)

one can roll all of the multiplcations by Hj and Hj' into a single level-3 call
by multiplying and adding various combinations of the dot products of the
columns of Hj. This function calculates the indices of the matrix of dot products,
that when multiplied together give the effective recursive action of Hj on H.

...
# Arguments
- `H::AbstractMatrix`: Apply the reflectors to this matrix
- `work::NameTuple`: namedtuple of work arrays
...

# Example
```julia
```
"""
function hotloop!(H::AbstractMatrix, work)
  Hj, y, dots, coeffs = work.Hj, work.y, work.dots, work.coeffs

  @assert size(H, 1) == size(work.Hj, 1)
  @assert size(H, 2) == size(work.y, 1)
  @assert size(work.Hj, 2) == size(work.y, 2)
  mul!(dots, Hj', Hj, true, false)
  mul!(y, H', Hj, true, false)

  # Collect all coefficients into a matrix using direct recurrence
  # coeffs[i, j] represents how much Hj' Hj contributes to H.
  # Starting with identity (each column contributes to itself)
  fill!(coeffs, 0)

  # Build coefficients column by column using recurrence relation
  # For each target column j, compute contributions from columns j+1:end
  @inbounds for j in 1:size(coeffs, 2)
    coeffs[j, j] = 1 # Column j contributes to itself
    # Each subsequent column i contributes based on previous contributions
    for i in j + 1:size(coeffs, 1)
      #for k in j:i - 1; coeffs[i, j] -= dots[i, k] * coeffs[k, j]; end
      ks = j:(i - 1)
      if !isempty(ks)
        mul!(view(coeffs, i:i, j:j), view(dots, i:i, ks), view(coeffs, ks, j), -1, true)
      end
    end
  end
  y' .= (coeffs * y') # these matrices are small
  mul!(H, work.Hj, work.y', -1, true) # H .-= Hj * y'
  return nothing
end

function householder!(H::MPIQRMatrix{T}, α=fill!(similar(H.localmatrix, size(H, 2)), 0); verbose=false,
    progress=FakeProgress()) where T
  m, n = size(H)
  @assert m >= n
  bs = blocksize(H) # the blocksize / tilesize of contiguous columns on each rank
  t1 = t2 = t3 = t4 = t5 = t6 = 0.0
  # work array for the BLAS call
  work = (Hj = similar(H.localmatrix, m, bs), # the H column(s)
          y = similar(H.localmatrix, localcolsize(H, 1:n), bs),
          dots = similar(H.localmatrix, bs, bs),
          coeffs = similar(H.localmatrix, bs, bs))

  @inbounds @views for j in 1:bs:n
    colowner = columnowner(H, j)
    bz = min(bs, n - j + 1)

    # process all the first bz column(s) of H
    if H.rank == colowner
      copyto!(work.Hj, view(H, :, j:j + bz - 1))
      @inbounds for Δj in 0:bz-1
        t1 += @elapsed s = norm(view(work.Hj, j + Δj:m, 1 + Δj)) # expensive
        t2 += @elapsed begin
          view(α, j + Δj) .= s .* alphafactor.(view(work.Hj, j + Δj, 1 + Δj))
          f = 1 ./ sqrt.(s .* (s .+ abs.(view(work.Hj, j + Δj, 1 + Δj))))
          view(work.Hj, j:j + Δj - 1, 1 + Δj) .= 0
          view(work.Hj, j + Δj, 1 + Δj) .-= view(α, j + Δj)
          view(work.Hj, j + Δj:m, 1 + Δj) .*= f
        end
        t3 += @elapsed hotloop!(view(work.Hj, j+Δj:m, 1 + Δj:bz),
                                (Hj=view(work.Hj, j+Δj:m, 1 + Δj),
                                 y=view(work.y,  1+Δj:bz),
                                 dots=view(work.dots, 1:1, 1:1), # indexing 1:1 dispatches to BLAS
                                 coeffs=view(work.coeffs, 1:1, 1:1))) # indexing 1:1 dispatches to BLAS
        t4 += @elapsed copyto!(view(H, j + Δj:m, j + Δj:j-1+bz), view(work.Hj, j + Δj:m, 1 + Δj:bz))
      end
    end
    t5 += @elapsed MPI.Bcast!(view(work.Hj, j:m, :), H.comm; root=colowner)

    # now do the next blocksize of colums to ready it for the next iteration
    t6 += @elapsed hotloop!(H, work, j:m, (j + bz):n)

    next!(progress)
  end
  ts = (t1, t2, t3, t4, t5, t6)
  verbose && println(sum(ts), "s: %s ", trunc.(100 .* ts ./ sum(ts), sigdigits=3))
  MPI.Allreduce!(α, +, H.comm)
  return MPIQRStruct(H, α)
end

"""
   solvedotu!(bi, H, b, i, n)

Does this but in a more efficient way
```julia
  @inbounds for j in intersect(H, i+1:n)
    @. bi += H[i, j] * b[j, :]
  end
```
"""
function solvedotu!(bi, H, b, i, n)
  indexa = searchsortedfirst(H.localcolumns, i+1)
  indexz = searchsortedlast(H.localcolumns, n)
  iszero(indexa:indexz) && return nothing # iszero slightly more flexible than isempty
  jiter = view(H.localcolumns, indexa:indexz)
  mul!(bi, transpose(view(b, jiter, :)), view(H.localmatrix, i, indexa:indexz))
  return nothing
end

function solve_householder!(b, H, α; progress=FakeProgress(), verbose=false)
  m, n = size(H)
  bs = blocksize(H)
  # multiply by Q' ...
  ta = tb = tc = td = te = 0.0
  @inbounds for j in 1:bs:n
    blockrank = columnowner(H, j)
    bs = min(bs, n - j + 1)
    if H.rank == blockrank
      for jj in 0:bs-1
        @assert columnowner(H, j) == blockrank
        viewbjj = view(b, j+jj:m, :) # use this view twice
        viewHjj = view(H, j+jj:m, j+jj) # use this view twice
        ta += @elapsed viewbjj .-= viewHjj .* (viewHjj' * viewbjj)
      end
    end
    tb += @elapsed MPI.Bcast!(view(b, j:m, :), H.comm; root=blockrank)
  end
  # now that b holds the value of Q'b
  # we may back sub with R
  @inbounds view(b, n, :) ./= view(α, n) # because iteration doesnt start at n
  bi = similar(b, size(b, 2))
  @inbounds @views for i in n-1:-1:1
    fill!(bi, 0)
    tc += @elapsed solvedotu!(bi, H, b, i, n)
    td += @elapsed MPI.Allreduce!(bi, +, H.comm)
    # axpby(α, x, β, y): !y .= x .* a .+ y .* β
    invαi = 1 ./ view(α, i)
    te += @elapsed axpby!(-invαi, bi, invαi, view(b, i, :)) # @. b[i, :] = (b[i, :] - bi) / α[i]
    next!(progress)
  end
  ts = (ta, tb, tc, td, te)
  verbose && println(sum(ts), " s: %s ", trunc.(100 .* ts ./ sum(ts), sigdigits=3))
  return b[1:n, :]
end

struct MPIQRStruct{T,M,Tα} <: AbstractMatrix{T}
  A::MPIQRMatrix{T,M}
  α::Tα
end

MPIQRStruct(A::MPIQRMatrix) = MPIQRStruct(A, fill!(similar(A.localmatrix, size(A, 2)), 0))
Base.size(s::MPIQRStruct) = size(s.A)
Base.size(s::MPIQRStruct, i::Integer) = size(s.A, i)
Base.setindex!(s::MPIQRStruct, v, i, j) = setindex!(s.A, v, i, j)
Base.getindex(s::MPIQRStruct, i, j) = getindex(s.A, i, j)

Progress(s::MPIQRStruct, dt=1; kwargs...) = Progress(s.A, dt=dt; kwargs...)
localcolumns(s::MPIQRStruct) = localcolumns(s.A)

function LinearAlgebra.qr!(H::MPIQRStruct; progress=FakeProgress(), verbose=false)
  householder!(H.A, H.α; progress=progress, verbose=verbose)
  return H
end

function LinearAlgebra.qr!(A::MPIQRMatrix; progress=FakeProgress(), verbose=false)
  H = MPIQRStruct(A)
  qr!(H; progress=progress, verbose=verbose)
  return H
end

function LinearAlgebra.ldiv!(H::MPIQRStruct, b::AbstractVecOrMat;
    progress=FakeProgress(), verbose=false)
  return solve_householder!(b, H.A, H.α; progress=progress, verbose=verbose)
end

function LinearAlgebra.ldiv!(x::AbstractVecOrMat, H::MPIQRStruct, b::AbstractVecOrMat;
    progress=FakeProgress(), verbose=false)
  c = deepcopy(b) # TODO: make this ...
  solve_householder!(c, H.A, H.α; progress=progress, verbose=verbose)
  x .= view(c, 1:size(x, 1), :) # .. and there this better
  return x
end

LinearAlgebra.:(\)(H::MPIQRStruct, b::AbstractVecOrMat) = ldiv!(H, deepcopy(b))

end

