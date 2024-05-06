module MPIQR

using LinearAlgebra, Random, Base.Threads
using MPI, Distributed, MPIClusterManagers

#man = MPIWorkerManager(2)

#addprocs(man)
#@mpi_do man begin
  Random.seed!(0)

  alphafactor(x::Real) = -sign(x)
  alphafactor(x::Complex) = -exp(im * angle(x))

  struct MPIMatrix{T} <: AbstractMatrix{T}
    localmatrix::Matrix{T}
    globalsize::Tuple{Int64, Int64}
    localcolumns::StepRange{Int, Int}
    rank::Int64
    comm::MPI.Comm
    commsize::Int64
    columnlookup::Vector{Int}
  end
  function MPIMatrix(localmatrix, globalsize, localcolumns, rank, comm, commsize)
    columnlookup = Vector{Int}([(x = searchsortedfirst(localcolumns, j); isnothing(x) ? 0 : x)  for j in 1:globalsize[2]])
    return MPIMatrix(localmatrix, globalsize, localcolumns, rank, comm, commsize, columnlookup)
  end
  Base.size(A::MPIMatrix) = A.globalsize
  function Base.getindex(A::MPIMatrix, i, j)
    A.localmatrix[i, A.columnlookup[j]]
  end
  function Base.setindex!(A::MPIMatrix, v::Number, i, j)
    return A.localmatrix[i, A.columnlookup[j]] = v
  end
  
  function householder!(A::AbstractMatrix{T}) where T
    m, n = size(A)
    H = A
    α = zeros(T, min(m, n)) # Diagonal of R
    Hj = zeros(T, m)
    t1 = t2 = t3 = t4 = t5 = 0.0
    t5 += @elapsed @inbounds @views for j in 1:n
      jinA = in(j, A.localcolumns)
      t1 += @elapsed if jinA
        s = norm(H[j:m, j])
        α[j] = s * alphafactor(H[j, j])
        f = 1 / sqrt(s * (s + abs(H[j, j])))
        H[j, j] -= α[j]
        H[j:m, j] .*= f
      end
      t2 += @elapsed jinArnk = MPI.Allreduce(jinA * H.rank, +, A.comm)
      t3 += @elapsed Hj[j:m] .= MPI.bcast(H[j:m, j], jinArnk, A.comm)
      t4 += @elapsed @views @threads for jj in intersect(A.localcolumns, j+1:n)
        s = sum(i->conj(Hj[i]) * H[i, jj], j:m)
        H[j:m, jj] .-= Hj[j:m] * s
      end
    end
    t6 = @elapsed α = MPI.Allreduce(α, +, A.comm)
    H.rank == 0 && @show t1, t2, t3, t4, t5, t6
    return (H, α)
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
      tc += @elapsed b1 .= MPI.Allreduce(b1, +, H.comm)
      b[j:m] .-= b1[j:m]
      b1[j:m] .= 0
    end
    # now that b holds the value of Q'b
    # we may back sub with R
    td += @elapsed MPI.Barrier(H.comm)
    @inbounds @views for i in n:-1:1
      bi = zero(eltype(b))
      te += @elapsed for j in i+1:n
        if in(j, H.localcolumns)
          bi += H[i, j] * b[j]
        end
      end
      tf += @elapsed bi = MPI.Allreduce(bi, +, H.comm)
      b[i] -= bi
      b[i] /= α[i]
    end
    H.rank == 0 && @show ta, tb, tc, td, te, tf
    return b[1:n]
  end

end

