# Utility
# -------
Base.:(==)(s::SNIrrep{N}, t::SNIrrep{N}) where {N} = (s.part == t.part)
Base.hash(s::SNIrrep, h::UInt) = hash(s.part, h)
Base.conj(s::SNIrrep) = s
Base.one(::Type{SNIrrep{N}}) where {N} = SNIrrep{N}([N])

sectorscalartype(::Type{<:S3Irrep}) = Float64
sectorscalartype(::Type{<:S4Irrep}) = Float64

TensorKitSectors.FusionStyle(::Type{S3Irrep}) = SimpleFusion()
TensorKitSectors.FusionStyle(::Type{S4Irrep}) = SimpleFusion()
TensorKitSectors.BraidingStyle(::Type{<:SNIrrep}) = Bosonic()

# Iterator over all allowed sectors
# custom implementation to keep state
# TODO: this is reversed...
struct SNIrrepValues{N}
    part::Vector{AbstractAlgebra.Generic.Partition{Int}}
    function SNIrrepValues{N}() where {N}
        return new{N}(sort!(AbstractAlgebra.Generic.partitions(N); rev=true))
    end
end
Base.values(::Type{SNIrrep{N}}) where {N} = SNIrrepValues{N}()

Base.IteratorEltype(::Type{<:SNIrrepValues}) = Base.HasEltype()
Base.eltype(::Type{SNIrrepValues{N}}) where {N} = SNIrrep{N}

Base.IteratorSize(::Type{<:SNIrrepValues}) = Base.HasLength()
Base.length(iter::SNIrrepValues) = length(iter.part)

function Base.iterate(iter::SNIrrepValues{N}, state...) where {N}
    next = iterate(iter.part, state...)
    isnothing(next) && return next
    p, nextstate = next
    return SNIrrep{N}(p), nextstate
end

Base.@propagate_inbounds function Base.getindex(iter::SNIrrepValues, i::Int)
    @boundscheck begin
        1 ≤ i ≤ length(iter) || throw(BoundsError(iter, i))
    end
    for (j, c) in enumerate(iter)
        j == i && return c
    end
    throw(BoundsError(iter, i))
end

function TensorKitSectors.findindex(iter::SNIrrepValues{N}, c::SNIrrep{N}) where {N}
    for (i, cc) in enumerate(iter)
        cc == c && return i
    end
    throw(ArgumentError(lazy"Cannot locate sector $c"))
end

# Fusion product
# --------------

function TensorKitSectors.:⊗(a::I, b::I) where {I<:SNIrrep}
    return Iterators.filter(c -> Nsymbol(a, b, c) > 0, values(I))
end

function fusiontensor(s1::I, s2::I, s3::I) where {I<:SNIrrep}
    return CGC(sectorscalartype(I), s1, s2, s3)
end

## Special cases with SimpleFusion

function Nsymbol(s1::I, s2::I, s3::I) where {I<:S3Irrep}
    c1, c2, c3 = 0, 0, 0
    for (a, s) in enumerate(values(S3Irrep))
        (s == s1) && (c1 = a)
        (s == s2) && (c2 = a)
        (s == s3) && (c3 = a)
        (c1 != 0) && (c2 != 0) && (c3 != 0) && break
    end
    ng = 6 # number of group elements in S3
    chis = char_tables.S3[c1, :] .* char_tables.S3[c2, :]
    return Bool(sum(ncs.S3 .* conj(char_tables.S3[c3, :]) .* chis) / ng)
end

function Nsymbol(s1::I, s2::I, s3::I) where {I<:S4Irrep}
    c1, c2, c3 = 0, 0, 0
    for (a, s) in enumerate(values(S4Irrep))
        (s == s1) && (c1 = a)
        (s == s2) && (c2 = a)
        (s == s3) && (c3 = a)
        (c1 != 0) && (c2 != 0) && (c3 != 0) && break
    end
    ng = 24 # number of group elements in S4
    chis = char_tables.S4[c1, :] .* char_tables.S4[c2, :]
    return Bool(sum(ncs.S4 .* conj(char_tables.S4[c3, :]) .* chis) / ng)
end

# SimpleFusion -> Fsymbol isa Number
function Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I<:Union{S3Irrep,S4Irrep}}
    (Nsymbol(a, b, e) && Nsymbol(e, c, d) && Nsymbol(b, c, f) && Nsymbol(a, f, d)) ||
        return zero(sectorscalartype(I))
    A = fusiontensor(a, b, e)[:, :, :, 1]
    B = fusiontensor(e, c, d)[:, :, 1, 1]
    C = fusiontensor(b, c, f)[:, :, :, 1]
    D = fusiontensor(a, f, d)[:, :, 1, 1]
    return @tensor conj(D[1, 5]) * conj(C[2, 4, 5]) * A[1, 2, 3] * B[3, 4]
end

# SimpleFusion -> Rsymbol isa Number
function Rsymbol(a::I, b::I, c::I) where {I<:Union{S3Irrep,S4Irrep}}
    (Nsymbol(a, b, c) && Nsymbol(b, a, c)) || return zero(sectorscalartype(I))
    A = fusiontensor(a, b, c)[:, :, 1, 1]
    B = fusiontensor(b, a, c)[:, :, 1, 1]
    return @tensor conj(B[1, 2]) * A[2, 1]
end

## GeneralFusion

function Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I<:SNIrrep}
    N1 = Nsymbol(a, b, e)
    N2 = Nsymbol(e, c, d)
    N3 = Nsymbol(b, c, f)
    N4 = Nsymbol(a, f, d)
    F = Array{sectorscalartype(I)}(undef, N1, N2, N3, N4)
    (N1 == 0 || N2 == 0 || N3 == 0 || N4 == 0) && return fill!(F, zero(sectorscalartype(I)))
    # computing first diagonal element
    A = fusiontensor(a, b, e)
    B = fusiontensor(e, c, d)[:, :, 1, :]
    C = fusiontensor(b, c, f)
    D = fusiontensor(a, f, d)[:, :, 1, :]
    @tensor F[-1, -2, -3, -4] =
        conj(D[1, 5, -4]) * conj(C[2, 4, 5, -3]) * A[1, 2, 3, -1] * B[3, 4, -2]
    return F
end

function Rsymbol(a::I, b::I, c::I) where {I<:SNIrrep}
    N1 = Nsymbol(a, b, c)
    N2 = Nsymbol(b, a, c)
    R = Array{sectorscalartype(I)}(undef, N1, N2)
    (N1 == 0 || N2 == 0) && return fill!(R, zero(sectorscalartype(I)))
    A = fusiontensor(a, b, c)[:, :, 1, :]
    B = fusiontensor(b, a, c)[:, :, 1, :]
    @tensor R[-1; -2] = conj(B[1, 2, -2]) * A[2, 1, -1]
    return R
end
