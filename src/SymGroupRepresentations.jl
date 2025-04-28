module SymGroupRepresentations

using LinearAlgebra
using AbstractAlgebra
using TensorKitSectors
using TensorOperations

export Sym, SNIrrep
export S3, S4, S5
export S3Irrep, S4Irrep, S5Irrep

using TensorKitSectors: Group, AbstractIrrep, IrrepTable
import TensorKitSectors: dim, Nsymbol, Fsymbol, Rsymbol, fusiontensor

# Group
# -----
abstract type Sym{N} <: Group end

const S3 = Sym{3}
const S4 = Sym{4}
const S5 = Sym{5}

# Irrep
# -----
"""
Construct the `S{N}` irrep with given partition of N
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
const S5Irrep = SNIrrep{5}

# Only S3, S4 have SimpleFusion
const SNIrrepSimple = Union{S3Irrep,S4Irrep}

Base.isless(s1::SNIrrep{N}, s2::SNIrrep{N}) where {N} = isless(s2.part, s1.part)

dim(s::SNIrrep) = Int(Generic.dim(YoungTableau(s.part)))

# generate CGC disk cache
include("gen_cache/irrep_data.jl")
include("gen_cache/projector.jl")

include("cgc.jl")
include("sector.jl")

const _allCGCs = (; :S3 => calall_CGCs(S3Irrep), :S4 => calall_CGCs(S4Irrep))
@info "CG coefficients for S3, S4 pre-calculated."

end
