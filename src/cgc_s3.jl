function CGC(::Type{T}, s1::S3Irrep, s2::S3Irrep, s3::S3Irrep) where {T}
    if isone(s1)
        @assert s2 == s3
        CGC = trivial_CGC(T, s2, true)
    elseif isone(s2)
        @assert s1 == s3
        CGC = trivial_CGC(T, s1, false)
    else
        CGC = zeros(T, dim(s1), dim(s2), dim(s3), 1)
        if s1 == s2 == S3Irrep([1, 1, 1])
            # [1₃] ⊗ [1₃] = [3]
            CGC[1, 1, 1, 1] = 1
        elseif s1 == S3Irrep([2, 1]) && s2 == S3Irrep([1, 1, 1])
            # [2₁1₁] ⊗ [1₃] = [2₁1₁]
            CGC[2, 1, 1, 1] = 1
            CGC[1, 1, 2, 1] = -1
        elseif s1 == S3Irrep([1, 1, 1]) && s2 == S3Irrep([2, 1])
            # [1₃] ⊗ [2₁1₁] = [2₁1₁]
            CGC[1, 2, 1, 1] = 1
            CGC[1, 1, 2, 1] = -1
        else
            # [2₁1₁] ⊗ [2₁1₁] = [3₁] + [2₁1₁] + [1₃]
            q = 1 / sqrt(2)
            if s3 == S3Irrep([3])
                CGC[1, 1, 1, 1] = q
                CGC[2, 2, 1, 1] = q
            elseif s3 == S3Irrep([2, 1])
                CGC[1, 2, 1, 1] = q
                CGC[2, 1, 1, 1] = q
                CGC[1, 1, 2, 1] = q
                CGC[2, 2, 2, 1] = -q
            else # s3 == S3Irrep([1, 1, 1])
                CGC[1, 2, 1, 1] = -q
                CGC[2, 1, 1, 1] = q
            end
        end
    end
    return CGC
end
