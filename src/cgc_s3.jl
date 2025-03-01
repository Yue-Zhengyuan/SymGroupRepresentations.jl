CGC(s1::I, s2::I, s3::I) where {I<:SNIrrep} = CGC(sectorscalartype(I), s1, s2, s3)

function CGC(::Type{T}, s1::S3Irrep, s2::S3Irrep, s3::S3Irrep) where {T}
    if isone(s1)
        @assert s2 == s3
        CGC = trivial_CGC(T, s2, true)
    elseif isone(s2)
        @assert s1 == s3
        CGC = trivial_CGC(T, s1, false)
    else
        CGC = zeros(T, dim(s1), sim(s2), dim(s3), 1)
        if s1 == s2 == S3Irrep([1, 1, 1])
            # [1₃] ⊗ [1₃] = [3]
            CGC[1, 1, 1, 1] = one(T)
        elseif s1 == S3Irrep([2, 1]) && s1 == S3Irrep([1, 1, 1])
            # [2₁1₁] ⊗ [1₃]
            CGC[1, 1, 2, 1] = one(T)
            CGC[2, 1, 1, 1] = -one(T)
        elseif s1 == S3Irrep([1, 1, 1]) && s1 == S3Irrep([2, 1])
            # [1₃] ⊗ [2₁1₁]
            CGC[1, 1, 2, 1] = one(T)
            CGC[1, 2, 1, 1] = -one(T)
        else
            # [2₁1₁] ⊗ [2₁1₁]
            q = 1 / sqrt(2)
            CGC[2, 1, 1, 1] = q
            CGC[2, 2, 1, 1] = q
            CGC[1, 1, 2, 1] = -q
            CGC[1, 2, 2, 1] = q
            CGC[1, 1, 3, 1] = q
            CGC[1, 2, 3, 1] = q
            CGC[2, 1, 4, 1] = -q
            CGC[2, 2, 4, 1] = q
        end
    end
    return CGC
end

# special case for 1 ⊗ s -> s or s ⊗ 1 -> s
function trivial_CGC(::Type{T}, s::SNIrrep, isleft::Bool=true) where {T<:Real}
    d = dim(s)
    CGC = zeros(T, d, d)
    @inbounds for m in axes(CGC, 1)
        CGC[m, m] = one(T)
    end
    return isleft ? reshape(CGC, 1, d, d, 1) : reshape(CGC, d, 1, d, 1)
end
