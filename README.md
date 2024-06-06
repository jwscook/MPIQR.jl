# QRMPI.jl: QR factorisation distributed over MPI

QR factorise your MPI distributed matrix using Householder reflections and then solve your square or least square problem.

Currently this only works for `Complex32`, `Float64`, `ComplexF32` and `ComplexF64` matrix and vector `eltype`s.
These `isbitstypes` make use of calls to `BLAS.gemv` and `BLAS.ger!` for highest performance, which is not far off threaded `LAPACK` `getrf` speeds.

An example:

```julia
using MPIQR
using LinearAlgebra, MPI, Distributed, MPIClusterManagers
using ProgressMeter # optional

MPI.Init(;threadlevel=MPI.THREAD_SERIALIZED)
const rnk = MPI.Comm_rank(MPI.COMM_WORLD)

function run(T=ComplexF64;)
  blocksize = 2
  m, n = 2048, 1024
  A0 = zeros(T, 0, 0)
  x1 = b0 = zeros(T, 0)
  if rnk == 0 # assemble and solve serially to compare with MPIQR later
    A0 = rand(T, m, n) # the original matrix
    b0 = rand(T, m) # the original lhs
    A1 = deepcopy(A0) # this will get mutated
    b1 = deepcopy(b0) # as will this
    x1 = qr!(A1) \ b1
    y1 = A0 * x1 # this is the matrix vector product, not the least squares solution
  end
  Aall = MPI.bcast(A0, 0, MPI.COMM_WORLD) # lhs matrix on all ranks
  ball = MPI.bcast(b0, 0, MPI.COMM_WORLD) # rhs vector on all ranks
  xall = MPI.bcast(x1, 0, MPI.COMM_WORLD) # solution vector on all ranks

  # get the columns of the matrix that will be local to this rank
  localcols = MPIQR.localcolumns(rnk, n, blocksize, MPI.Comm_size(MPI.COMM_WORLD))
  b = deepcopy(ball)

  # distribute the serial matrix onto the columns local to this rank
  A = MPIQR.MPIQRMatrix(deepcopy(Aall[:, localcols]), size(Aall); blocksize=blocksize)
  y2 = A * xall # make sure matrix vector multiplication works...
  if iszero(rnk) # ... and is correct.
    @assert y2 ≈ y1
  end

  # qr! optionally accepts a progress meter
  progress = Progress(size(A, 2) ÷ blocksize, dt=1; showspeed=true)
  x2 = qr!(A; progress=progress) \ b # qr factorize A in-place and solve

  if iszero(rnk) # now see if the answer is right...
    @assert norm(Aall' * Aall * xall .- Aall' * ball) < 1e-8
    @show residual = norm(Aall' * Aall * x2 .- Aall' * ball)
  end
end
run()

MPI.Finalize()

```
