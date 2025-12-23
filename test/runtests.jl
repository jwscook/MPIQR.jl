using MPIQR

using LinearAlgebra, Random, Base.Threads, Test
using MPI, Distributed, MPIClusterManagers

MPI.Init(;threadlevel=MPI.THREAD_SERIALIZED)
const cmm = MPI.COMM_WORLD
const rnk = MPI.Comm_rank(cmm)
const sze = MPI.Comm_size(cmm)
const nts = Threads.nthreads()

BLAS.set_num_threads(nts)

using Random, ProgressMeter
function run(blocksizes=(1,2,3,4), npows=(8,10,), Ts=(ComplexF64,); bestof=4)
  for npow in npows, T in Ts
    Random.seed!(0)
    n = 2^npow
    m = n + 2^(npow-2)
    A0 = rand(T, m, n)
    b0 = rand(T, m, 2)
    x1 = zeros(T, 0)
    if rnk == 0
      BLAS.set_num_threads(nts * sze)
      A1 = deepcopy(A0)
      b1 = deepcopy(b0)
      x1 = qr!(A1, NoPivot()) \ b1
      y1 = A0 * x1
      t1s = []
      for _ in 1:bestof
        push!(t1s, @elapsed qr!(A1, NoPivot()) \ b1)
      end
      t1 = minimum(t1s)
      BLAS.set_num_threads(nts)
    end
    x1 = MPI.bcast(x1, 0, cmm)

    for blocksize in blocksizes
      localcols = MPIQR.localcolumns(rnk, n, blocksize, sze)
      b = deepcopy(b0)

      A = MPIQR.MPIQRMatrix(deepcopy(A0[:, localcols]), size(A0); blocksize=blocksize)
      y2 = A * x1
      if iszero(rnk)
        @test y2 ≈ y1
      end
      MPI.Barrier(cmm)
      #x2 = qr!(A) \ b
      dt = iszero(rnk) ? 1 : 2^31
      x2 = similar(x1)
      qrA = qr!(A; verbose=false, progress=Progress(A, dt=dt; showspeed=true))
      ldiv!(x2, qrA, b; verbose=false, progress=Progress(A, dt=dt/10; showspeed=true))
      x3 = qrA \ b
      localcols = MPIQR.localcolumns(qrA)
      if !isempty(localcols)
      qrA.A.localmatrix .= A0[:, localcols] # re-use it
      @test all(qrA[:, localcols] .== A0[:, localcols]) # re-use it
      fill!(qrA.A.localmatrix, -1) # fill with distintive value and test we overwrite it
      qrA[:, localcols] .= A0[:, localcols] # re-use it
      @test x2 ≈ x3
      @test all(qrA[:, localcols] .== A0[:, localcols]) # re-use it
      qrA[1, localcols[1]] = π
      @test qrA[1, localcols[1]] == T(π)
      end

      t2s = []
      for _ in 1:bestof
        push!(t2s, @elapsed begin
          progress = Progress(A, dt=dt; showspeed=true)
          qr!(A; progress=progress) \ b
        end)
      end
      t2 = minimum(t2s)

      MPI.Barrier(cmm)

      if iszero(rnk)
          @test norm(A0' * A0 * x1 .- A0' * b0) < 1e-8
          res = norm(A0' * A0 * x2 .- A0' * b0)
          try
            println("np=$sze, nt=$nts, T=$T, m=$m, n=$n, blocksize=$blocksize:")
            println("tLAPACK = $t1, tMPIQR = $t2, ratio=$(t2/t1)x")
            @test res < 1e-8
            println("    PASSED: norm of residual = $res")
          catch
            println("    FAILED: norm of residual = $res")
          end
      end
    end
  end
end
run()

MPI.Finalize()

