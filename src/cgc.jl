function CGC(::Type{T}, s1::R, s2::R, s3::R) where {T,R<:SNIrrep}
    ndeg = _Nsymbol(s1, s2, s3)
    cgc = zeros(T, dim(s1), dim(s2), dim(s3), ndeg)
    c1, c2, c3 = 0, 0, 0
    for (a, s) in enumerate(values(R))
        (s == s1) && (c1 = a)
        (s == s2) && (c2 = a)
        (s == s3) && (c3 = a)
        (c1 != 0) && (c2 != 0) && (c3 != 0) && break
    end
    if R == S3Irrep
        allcgcs = _allCGCs.S3
    elseif R == S4Irrep
        allcgcs = _allCGCs.S4
    elseif R == S5Irrep
        allcgcs = _allCGCs.S5
    else
        error("$R is not implemented.")
    end
    for idx in CartesianIndices(cgc)
        i1, i2, i3, deg = Tuple(idx)
        key = (c1 <= c2) ? (c1, c2, c3, i1, i2, i3, deg) : (c2, c1, c3, i2, i1, i3, deg)
        if haskey(allcgcs, key)
            cgc[idx] = allcgcs[key]
        end
    end
    return cgc
end

CGC(s1::I, s2::I, s3::I) where {I<:SNIrrep} = CGC(sectorscalartype(I), s1, s2, s3)
