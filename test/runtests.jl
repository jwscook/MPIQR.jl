using MPIQR

using LinearAlgebra, Random, Base.Threads
using MPI, Distributed, MPIClusterManagers

MPI.Init()
const cmm = MPI.COMM_WORLD
const rnk = MPI.Comm_rank(cmm)
const sze = MPI.Comm_size(cmm)

#include("qrmpi.jl")

Random.seed!(0)

for mn in ((60, 50), (600, 500), (1200, 1000), (2400, 2000)), T in (Float64, ComplexF64)
  m, n = mn
  iszero(rnk) && @show T, m, n
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

  localcols = (rnk + 1:sze:size(Aall, 2))
  b = deepcopy(ball)

  A = MPIQR.MPIMatrix(deepcopy(Aall[:, localcols]), size(Aall), localcols, rnk, cmm, sze)

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
