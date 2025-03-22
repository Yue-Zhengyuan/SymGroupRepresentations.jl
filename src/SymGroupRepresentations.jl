module SymGroupRepresentations

using LinearAlgebra
using AbstractAlgebra
using TensorKitSectors
using TensorOperations

export Sym, SNIrrep
export S3, S4
export S3Irrep, S4Irrep

using TensorKitSectors: Group, AbstractIrrep, IrrepTable
import TensorKitSectors: dim, Nsymbol, Fsymbol, Rsymbol, fusiontensor

# Group
# -----
abstract type Sym{N} <: Group end

const S3 = Sym{3}
const S4 = Sym{4}

# Irrep
# -----
"""
Construct the S{N} irrep with given partition of N
"""
struct SNIrrep{N} <: AbstractIrrep{Sym{N}}
    part::Generic.Partition{Int}

    function SNIrrep{N}(part::Generic.Partition) where {N}
        (sum(part) == N) ||
            throw(ArgumentError("Must provide partition of $N for `SNIrrep{$N}`"))
        return new{N}(part)
    end
end

function SNIrrep{N}(part::Vector{I}) where {N,I<:Integer}
    return SNIrrep{N}(Partition(part))
end
function SNIrrep(part::Vector{I}) where {I<:Integer}
    return SNIrrep{sum(part)}(Partition(part))
end

const S3Irrep = SNIrrep{3}
const S4Irrep = SNIrrep{4}

Base.isless(s1::SNIrrep{N}, s2::SNIrrep{N}) where {N} = isless(s2.part, s1.part)

dim(s::SNIrrep) = Int(Generic.dim(YoungTableau(s.part)))

include("cgc.jl")
include("cgc_s3.jl")
include("cgc_s4.jl")
include("sector.jl")
include("sector_s3.jl")
include("sector_s4.jl")

end
