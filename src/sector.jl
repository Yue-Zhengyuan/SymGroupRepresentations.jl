# Utility
# -------
Base.:(==)(s::SNIrrep{N}, t::SNIrrep{N}) where {N} = (s.part == t.part)
Base.isless(s::SNIrrep{N}, t::SNIrrep{N}) where {N} = isless(s.part, t.part)
Base.hash(s::SNIrrep, h::UInt) = hash(s.part, h)
Base.conj(s::SNIrrep) = s
Base.one(::Type{SNIrrep{N}}) where {N} = SNIrrep{N}([N])

TensorKitSectors.sectorscalartype(::Type{<:S3Irrep}) = Float64

TensorKitSectors.FusionStyle(::Type{S3Irrep}) = SimpleFusion() # no multiplicity
TensorKitSectors.BraidingStyle(::Type{<:SNIrrep}) = Bosonic()

# Iterator over all allowed sectors
# custom implementation to keep state
# TODO: verify that it isn't a problem that this iterator is stateful
struct SNIrrepValues{N}
    part::AllParts
    SNIrrepValues{N}() = new{AllParts(N)}
end
Base.values(::Type{SNIrrep{N}}) where {N} = SNIrrepValues{N}()

Base.IteratorEltype(::Type{<:SNIrrepValues}) = HasEltype()
Base.eltype(::Type{SNIrrepValues{N}}) where {N} = SNIrrep{N}

Base.IteratorSize(::Type{<:SNIrrepValues}) = HasLength()
Base.length(iter::SNIrrepValues{N}) where {N} = length(iter.part)

function Base.iterate(iter::SNIrrepValues, state...) where {N}
    next = iterate(iter, state...)
    isnothing(next) && return next
    p, nextstate = next
    return SNIrrep{N}(p), nextstate
end

# Fusion product
# --------------
function TensorKitSectors.:⊗(s1::S3Irrep, s2::S3Irrep)
    # since s1 ⊗ s2 = s2 ⊗ s1, we assume s1 > s2
    s1 < s2 && return (s2 ⊗ s1)
    if s1 == S3Irrep([3]) # trivial rep
        return (s2,)
    elseif s1 == S3Irrep([2, 1])
        if s2 == S3Irrep([1, 1, 1])
            return (s1,)
        else
            return (S3Irrep([1, 1, 1]), S3Irrep([2, 1]), S3Irrep([3]))
        end
    else # s1 = s2 = S3Irrep([1, 1, 1])
        return (S3Irrep([3]),)
    end
end

function TensorKitSectors.Nsymbol(s1::S3Irrep, s2::S3Irrep, s3::S3Irrep)
    return (s3 in (s1 ⊗ s2))
end

function TensorKitSectors.fusiontensor(s1::I, s2::I, s3::I) where {I<:SNIrrep}
    return CGC(sectorscalartype(I), s1, s2, s3)
end

function TensorKitSectors.Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I<:SNIrrep}
    N1 = Nsymbol(a, b, e)
    N2 = Nsymbol(e, c, d)
    N3 = Nsymbol(b, c, f)
    N4 = Nsymbol(a, f, d)
    (N1 == 0 || N2 == 0 || N3 == 0 || N4 == 0) &&
        return fill(zero(sectorscalartype(I)), N1, N2, N3, N4)
    # computing first diagonal element
    A = fusiontensor(a, b, e)
    B = fusiontensor(e, c, d)[:, :, 1, :]
    C = fusiontensor(b, c, f)
    D = fusiontensor(a, f, d)[:, :, 1, :]
    @tensor F[-1, -2, -3, -4] :=
        conj(D[1, 5, -4]) * conj(C[2, 4, 5, -3]) * A[1, 2, 3, -1] * B[3, 4, -2]
    return Array(F)
end

function TensorKitSectors.Rsymbol(a::I, b::I, c::I) where {I<:SNIrrep}
    N1 = Nsymbol(a, b, c)
    N2 = Nsymbol(b, a, c)
    (N1 == 0 || N2 == 0) && return fill(zero(sectorscalartype(I)), N1, N2)
    A = fusiontensor(a, b, c)[:, :, 1, :]
    B = fusiontensor(b, a, c)[:, :, 1, :]
    @tensor R[-1; -2] := conj(B[1, 2, -2]) * A[2, 1, -1]
    return Array(R)
end
