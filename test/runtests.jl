using Test
using TestExtras
using Random
using TensorKitSectors
using SymGroupRepresentations
using TensorOperations
using Base.Iterators: take, product
using LinearAlgebra: LinearAlgebra

const TKS = TensorKitSectors

include("testsetup.jl")
using .TestSetup

const sectorlist = (S3Irrep,)

@testset "$(TensorKitSectors.type_repr(I))" for I in sectorlist
    @include("sectors.jl")
    include("fusiontrees.jl")
end

@testset "Deligne product" begin
    sectorlist′ = (Trivial, sectorlist...)
    for I1 in sectorlist′, I2 in sectorlist′
        a = first(smallset(I1))
        b = first(smallset(I2))

        @constinferred a ⊠ b
        @constinferred a ⊠ b ⊠ a
        @constinferred a ⊠ b ⊠ a ⊠ b
        @constinferred I1 ⊠ I2
        @test typeof(a ⊠ b) == I1 ⊠ I2
    end
end

@testset "Aqua" begin
    using Aqua: Aqua
    Aqua.test_all(TensorKitSectors)
end

@testset "JET" begin
    using JET: JET
    JET.test_package(TensorKitSectors; target_defined_modules=true)
end
