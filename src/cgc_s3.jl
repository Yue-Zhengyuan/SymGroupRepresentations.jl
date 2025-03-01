TensorKitSectors.FusionStyle(::Type{S3Irrep}) = SimpleFusion()

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

CGC(s1::I, s2::I, s3::I) where {I<:SNIrrep} = CGC(Float64, s1, s2, s3)

function CGC(::Type{T}, s1::S3Irrep, s2::S3Irrep, s3::S3Irrep) where {T}
    if isone(s1)
        @assert s2 == s3
        CGC = trivial_CGC(T, s2, true)
    elseif isone(s2)
        @assert s1 == s3
        CGC = trivial_CGC(T, s1, false)
    else
        CGC = Array{T}(undef, dim(s1), sim(s2), dim(s3), 1)
        if s1 == s2 == S3Irrep([1, 1, 1])
            # [1₃] ⊗ [1₃] = [3]
            CGC[1, 1, 1, 1] = 1.0
        elseif s1 == S3Irrep([2, 1]) && s1 == S3Irrep([1, 1, 1])
            # [2₁1₁] ⊗ [1₃]
            CGC[1, 1, 2, 1] = 1.0
            CGC[2, 1, 1, 1] = -1.0
        elseif s1 == S3Irrep([1, 1, 1]) && s1 == S3Irrep([2, 1])
            # [1₃] ⊗ [2₁1₁]
            CGC[1, 1, 2, 1] = 1.0
            CGC[1, 2, 1, 1] = -1.0
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
function trivial_CGC(::Type{T}, s::SNIrrep, isleft=true) where {T<:Real}
    d = dim(s)
    if isleft
        CGC = Array{T}(undef, 1, d, d, 1)
        for m in 1:d
            CGC[1, m, m, 1] = one(T)
        end
    else
        CGC = Array{T}(undef, d, 1, d, 1)
        for m in 1:d
            CGC[m, 1, m, 1] = one(T)
        end
    end
    return CGC
end
