using LinearAlgebra, Test, Base.Threads
using Combinatorics


@testset "hot loop logic" begin

T = ComplexF64
A0 = rand(T, 6, 5)
H1 = rand(T, 6)
H2 = rand(T, 6)
H3 = rand(T, 6)
H4 = rand(T, 6)
H5 = rand(T, 6)

Hs = (H1, H2, H3, H4, H5)
for i in eachindex(Hs)
  Hs[i][1:i] .= 0
end

A1 = A0 - H1 * H1' * A0
A2 = A1 - H2 * H2' * A1
A3 = A2 - H3 * H3' * A2
A4 = A3 - H4 * H4' * A3
A5 = A4 - H5 * H5' * A4

As = deepcopy.((A1, A2, A3, A4, A5))


function ladders(N, A=0)
  A >= N && return Any[(N, N)]
  output = Any[(A, N)]
  for i in 1:N-1, c in combinations(A+1:N-1, i)
    push!(output, (A, c..., N))
  end
  return reverse(output)
end

function dotproducts(Hj)
  dots = Dict()
  @views @inbounds for i in 1:size(Hj, 2), j in 1:i
    dots[(i, j)] = dot(Hj[:, i], Hj[:, j])
  end
  return dots
end

for h in 1:5
  Hj = ([H1 H2 H3 H4 H5])[:, 1:h]

  dots = dotproducts(Hj)
  Hc = deepcopy(Hj)

  @views @inbounds for ii in 0:size(Hj, 2)-1
    for i in ii+1:size(Hj, 2)-1
      for ladder in ladders(i, ii)
        factor = prod(dots[(ladder[j] + 1, ladder[j-1] + 1)] for j in 2:length(ladder))
        sgn = -(-1)^length(ladder)
        factor *= sgn
        BLAS.axpy!(factor, view(Hc, :, i + 1), view(Hc, :, ii + 1))
      end
    end
  end

  y = zeros(T, size(A0, 2), size(Hj, 2))
  mul!(y, A0', Hj, true, false) # y = A0' * Hj
  #@show size(y), size(A0), size(Hj)
  AN = deepcopy(A0)
  #AN -= Hc * y'
  BLAS.gemm!('N', 'C', -one(T), Hc, y, true, AN) #Update C as alpha*A*B + beta*C
  @test AN ≈ As[size(Hj, 2)]
end
end

