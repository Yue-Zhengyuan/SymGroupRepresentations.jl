CGC(s1::I, s2::I, s3::I) where {I<:SNIrrep} = CGC(sectorscalartype(I), s1, s2, s3)

# special case for 1 ⊗ s -> s or s ⊗ 1 -> s
function trivial_CGC(::Type{T}, s::SNIrrep, isleft::Bool=true) where {T<:Real}
    d = dim(s)
    CGC = zeros(T, d, d)
    @inbounds for m in axes(CGC, 1)
        CGC[m, m] = one(T)
    end
    return isleft ? reshape(CGC, 1, d, d, 1) : reshape(CGC, d, 1, d, 1)
end
