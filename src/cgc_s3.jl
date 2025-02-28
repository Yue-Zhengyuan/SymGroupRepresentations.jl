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
    return _get_CGC(T, (s1, s2, s3))
end

# special case for 1 ⊗ s -> s or s ⊗ 1 -> s
function trivial_CGC(::Type{T}, s::SUNIrrep, isleft=true) where {T<:Real}
    d = dim(s)
    if isleft
        CGC = SparseArray{T}(undef, 1, d, d, 1)
        for m in 1:d
            CGC[1, m, m, 1] = one(T)
        end
    else
        CGC = SparseArray{T}(undef, d, 1, d, 1)
        for m in 1:d
            CGC[m, 1, m, 1] = one(T)
        end
    end
    return CGC
end
