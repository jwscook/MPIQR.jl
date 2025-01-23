using MPIQR
using Test

@testset "MPIQR unit tests" begin
  for commsize in 1:8, blocksize in 1:8, n in 1:128
    alllocalcols = []
    for rnk in 0:commsize-1
      push!(alllocalcols, MPIQR.localcolumns(rnk, n, blocksize, commsize))
    end
    @test sort(vcat(alllocalcols...)) == 1:n
  end
end
