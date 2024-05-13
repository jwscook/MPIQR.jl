using MPIQR

using LinearAlgebra, Random, Base.Threads, Test
using MPI, Distributed, MPIClusterManagers

MPI.Init(;threadlevel=MPI.THREAD_SERIALIZED)
const cmm = MPI.COMM_WORLD
const rnk = MPI.Comm_rank(cmm)
const sze = MPI.Comm_size(cmm)
const nts = Threads.nthreads()

BLAS.set_num_threads(nts)

using Random

@static if Sys.islinux()
  using ThreadPinning
end

function run()
  for blocksize in (1, 2, 3, 4), npow in 8:2:12, T in (ComplexF64, )#Float64,
    @static if Sys.islinux()
      cpus = rnk * nts:(rnk + 1) * nts
      ThreadPinning.pinthreads(cpus)
    end

    Random.seed!(0)
    n = (2^npow ÷ blocksize) * blocksize
    m = n + 2^(npow-2)
    A0 = zeros(T, 0, 0)
    x1 = b0 = zeros(T, 0)
    if rnk == 0
      BLAS.set_num_threads(nts * sze)
      A0 = rand(T, m, n)
      b0 = rand(T, m)
      A1 = deepcopy(A0)
      b1 = deepcopy(b0)
      x1 = qr!(A1, NoPivot()) \ b1
      t1s = []
      for _ in 1:3
        push!(t1s, @elapsed qr!(A1, NoPivot()) \ b1)
      end
      t1 = minimum(t1s)
      BLAS.set_num_threads(nts)
    end
    Aall = MPI.bcast(A0, 0, cmm)
    ball = MPI.bcast(b0, 0, cmm)
    xall = MPI.bcast(x1, 0, cmm)
    x1 = xall

    localcols = MPIQR.localcolumns(rnk, n, blocksize, sze)
    b = deepcopy(ball)

    A = MPIQR.MPIQRMatrix(deepcopy(Aall[:, localcols]), size(Aall); blocksize=blocksize)

    MPI.Barrier(cmm)
    H, α = MPIQR.householder!(A)
    x2 = MPIQR.solve_householder!(b, H, α)

    t2s = []
    for _ in 1:3
      push!(t2s, @elapsed begin
        H, α = MPIQR.householder!(A)
        MPIQR.solve_householder!(b, H, α)
      end)
    end
    t2 = minimum(t2s)

    MPI.Barrier(cmm)

    if iszero(rnk)
        @assert norm(Aall' * Aall * x1 .- Aall' * ball) < 1e-8
        res = norm(Aall' * Aall * x2 .- Aall' * ball)
        try
          println("np=$sze, nt=$nts, T=$T, m=$m, n=$n, blocksize=$blocksize took $(t2/t1)x")
          @assert res < 1e-8
          println("    PASSED: norm of residual = $res")
        catch
          println("    FAILED: norm of residual = $res")
        end
    end
  end
end
run()

MPI.Finalize()
