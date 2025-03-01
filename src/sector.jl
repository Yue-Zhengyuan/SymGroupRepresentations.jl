Base.:(==)(s::SNIrrep, t::SNIrrep) = (s.part == t.part)
Base.hash(s::SNIrrep, h::UInt) = hash(s.part, h)
Base.conj(s::SNIrrep) = s
Base.one(::Type{SNIrrep{N}}) where {N} = SNIrrep{N}([N])

TensorKitSectors.BraidingStyle(::Type{<:SNIrrep}) = Bosonic()
Base.isreal(::Type{<:SNIrrep}) = true

function TensorKitSectors.fusiontensor(
    s1::SNIrrep{N}, s2::SNIrrep{N}, s3::SNIrrep{N}
) where {N}
    return CGC(Float64, s1, s2, s3)
end

function TensorKitSectors.Fsymbol(
    a::SNIrrep{N}, b::SNIrrep{N}, c::SNIrrep{N}, d::SNIrrep{N}, e::SNIrrep{N}, f::SNIrrep{N}
) where {N}
    N1 = Nsymbol(a, b, e)
    N2 = Nsymbol(e, c, d)
    N3 = Nsymbol(b, c, f)
    N4 = Nsymbol(a, f, d)
    (N1 == 0 || N2 == 0 || N3 == 0 || N4 == 0) && return fill(0.0, N1, N2, N3, N4)
    # computing first diagonal element
    A = fusiontensor(a, b, e)
    B = fusiontensor(e, c, d)[:, :, 1, :]
    C = fusiontensor(b, c, f)
    D = fusiontensor(a, f, d)[:, :, 1, :]
    @tensor F[-1, -2, -3, -4] :=
        conj(D[1, 5, -4]) * conj(C[2, 4, 5, -3]) * A[1, 2, 3, -1] * B[3, 4, -2]
    return Array(F)
end

function TensorKitSectors.Rsymbol(a::SNIrrep{N}, b::SNIrrep{N}, c::SNIrrep{N}) where {N}
    N1 = Nsymbol(a, b, c)
    N2 = Nsymbol(b, a, c)
    (N1 == 0 || N2 == 0) && return fill(0.0, N1, N2)
    A = fusiontensor(a, b, c)[:, :, 1, :]
    B = fusiontensor(b, a, c)[:, :, 1, :]
    @tensor R[-1; -2] := conj(B[1, 2, -2]) * A[2, 1, -1]
    return Array(R)
end
