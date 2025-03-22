function CGC(::Type{T}, s1::S4Irrep, s2::S4Irrep, s3::S4Irrep) where {T}
    if isone(s1)
        @assert s2 == s3
        CGC = trivial_CGC(T, s2, true)
    elseif isone(s2)
        @assert s1 == s3
        CGC = trivial_CGC(T, s1, false)
    else
        CGC = zeros(T, dim(s1), dim(s2), dim(s3), 1)
        if s1 == s2 == S4Irrep([1, 1, 1, 4])
            CGC[1, 1, 1, 1] = 1
        elseif s1 == S4Irrep()
        end
    end
    return CGC
end