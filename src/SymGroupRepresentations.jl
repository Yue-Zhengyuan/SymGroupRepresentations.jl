module SymGroupRepresentations

using LinearAlgebra
using AbstractAlgebra
using TensorKitSectors
using TensorOperations

export Sym, SNIrrep
export S3, S4
export S3Irrep, S4Irrep

abstract type Sym{N} <: TensorKitSectors.Group end

const S3 = Sym{3}
const S4 = Sym{4}

"""
Construct the S{N} irrep with given partition of 3
"""
struct SNIrrep{N} <: AbstractIrrep{Sym{N}}
    part::Generic.Partition

    function SNIrrep{N}(part::Generic.Partition) where {N}
        (sum(part) == N) ||
            throw(ArgumentError("Must provide partition of $N for `SNIrrep{$N}`"))
        return new{N}(part)
    end
end

function SNIrrep{N}(part::Vector{I}) where {N,I<:Integer}
    return SNIrrep{sum(part)}(Partition(part))
end
function SNIrrep(part::Vector{I}) where {I<:Integer}
    return SNIrrep{sum(part)}(Partition(part))
end

const S3Irrep = SNIrrep{3}
const S4Irrep = SNIrrep{4}

function Base.isless(s1::SNIrrep{N}, s2::SNIrrep{N}) where {N}
    return s1.part < s2.part
end

function TensorKitSectors.dim(s::SNIrrep)
    return Generic.dim(YoungTableau(s.part))
end

include("cgc_s3.jl")
include("sector.jl")

end
