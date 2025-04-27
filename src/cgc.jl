# special case for 1 ⊗ s -> s or s ⊗ 1 -> s
function trivial_CGC(::Type{T}, s::SNIrrep, isleft::Bool=true) where {T<:Real}
    d = dim(s)
    CGC = zeros(T, d, d)
    @inbounds for m in axes(CGC, 1)
        CGC[m, m] = one(T)
    end
    return isleft ? reshape(CGC, 1, d, d, 1) : reshape(CGC, d, 1, d, 1)
end

function CGC(::Type{T}, s1::R, s2::R, s3::R) where {T,R<:SNIrrep}
    cgc = zeros(T, dim(s1), dim(s2), dim(s3), 1)
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
    else
        error("CG coefficients are not implemented for $R.")
    end
    for idx in CartesianIndices(cgc)
        i1, i2, i3, _ = Tuple(idx)
        key = (c1, c2, c3, i1, i2, i3)
        if haskey(allcgcs, key)
            cgc[idx] = allcgcs[key]
        end
    end
    return cgc
end

CGC(s1::I, s2::I, s3::I) where {I<:SNIrrep} = CGC(sectorscalartype(I), s1, s2, s3)
