using MPIQR

using LinearAlgebra, Random, Base.Threads
using MPI, Distributed, MPIClusterManagers

MPI.Init(;threadlevel=MPI.THREAD_SERIALIZED)
const cmm = MPI.COMM_WORLD
const rnk = MPI.Comm_rank(cmm)
const sze = MPI.Comm_size(cmm)
const nts = Threads.nthreads()

BLAS.set_num_threads(nts)
@static if Sys.islinux()
  using ThreadPinning
  cpus = rnk * nts:(rnk + 1) * nts
  ThreadPinning.pinthreads(cpus)
end

#include("qrmpi.jl")
using Random
Random.seed!(0)

for blocksize in (1, 4), npow in 7:1:10, T in (ComplexF64, )#Float64, 
  n = 2^npow
  m = n + 2^(npow-2)
  iszero(rnk) && @show T, m, n, blocksize
  A0 = zeros(T, 0, 0)
  x1 = b0 = zeros(T, 0)
  if rnk == 0
    A0 = rand(T, m, n)
    b0 = rand(T, m)
    A1 = deepcopy(A0)
    b1 = deepcopy(b0)
    t1 = @elapsed x1 = qr!(A1, NoPivot()) \ b1
  end
  Aall = MPI.bcast(A0, 0, cmm)
  ball = MPI.bcast(b0, 0, cmm)
  xall = MPI.bcast(x1, 0, cmm)
  x1 = xall

  localcols = MPIQR.localcolumns(rnk, n, blocksize, sze)
  b = deepcopy(ball)

  A = MPIQR.MPIMatrix(deepcopy(Aall[:, localcols]), size(Aall); blocksize=blocksize)

  t2 = @elapsed  begin
    H, α = MPIQR.householder!(A)
    x2 = MPIQR.solve_householder!(b, H, α)
  end
  if iszero(rnk)
    @show norm(Aall' * Aall * x1 .- Aall' * ball)
    @show norm(Aall' * Aall * x2 .- Aall' * ball)
    @show t2 / t1
  end
end

MPI.Finalize()
