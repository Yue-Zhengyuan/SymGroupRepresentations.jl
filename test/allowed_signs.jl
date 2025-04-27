using Test
using LinearAlgebra
import SymGroupRepresentations: allowed_signs

for _ in 1:30
    d = 10
    A = rand(d, d)
    u0 = [rand() < 0.5 ? -1 : 1 for _ in 1:d]
    U0 = diagm(u0)
    B = U0 * A * U0
    # randomly set some elements to 0
    zero_mask = rand(size(A)...) .< 0.8
    nzero = sum(zero_mask)
    A[zero_mask] .= 0
    B[zero_mask] .= 0
    us = allowed_signs(A, B)
    @info "Set $nzero elements to 0. Find $(length(us)) possible basis signs."
    @test (u0[1] * u0) in us
    for u in us
        U = diagm(u)
        @test A ≈ U * B * U
    end
end
