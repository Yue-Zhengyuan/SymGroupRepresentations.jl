function TensorKitSectors.:⊗(s1::S4Irrep, s2::S4Irrep)
    # since s1 ⊗ s2 = s2 ⊗ s1, we assume s1 ≤ s2
    s1 > s2 && return (s2 ⊗ s1)
    if s1 == S4Irrep([4]) # trivial rep
        return [s2]
    elseif s1 == S4Irrep([3, 1])
        if s2 == s1
            return [S4Irrep([4]), S4Irrep([3, 1]), S4Irrep([2, 2]), S4Irrep([2, 1, 1])]
        elseif s2 == S4Irrep([2, 2])
            return [S4Irrep([3, 1]), S4Irrep([2, 1, 1])]
        elseif s2 == S4Irrep([2, 1, 1])
            return [
                S4Irrep([3, 1]), S4Irrep([2, 2]), S4Irrep([2, 1, 1]), S4Irrep([1, 1, 1, 1])
            ]
        else # s2 == S4Irrep([1,1,1,1])
            return [S4Irrep([2, 1, 1])]
        end
    elseif s1 == S4Irrep([2, 2])
        if s2 == s1
            return [S4Irrep([4]), S4Irrep([2, 2]), S4Irrep([1, 1, 1, 1])]
        elseif s2 == S4Irrep([2, 1, 1])
            return [S4Irrep([3, 1]), S4Irrep([2, 1, 1])]
        else # s2 == S4Irrep([1,1,1,1])
            return [S4Irrep([2, 2])]
        end
    elseif s1 == S4Irrep([2, 1, 1])
        if s2 == s1
            return [S4Irrep([4]), S4Irrep([3, 1]), S4Irrep([2, 2]), S4Irrep([2, 1, 1])]
        else # s2 == S4Irrep([1,1,1,1])
            return [S4Irrep([3, 1])]
        end
    else # s1 = s2 = S4Irrep([1, 1, 1, 1])
        return [S4Irrep([4])]
    end
end

function Nsymbol(s1::S4Irrep, s2::S4Irrep, s3::S4Irrep)
    N = (s3 in (s1 ⊗ s2))
    @assert !ismissing(N)
    return N
end

# S3 has SimpleFusion -> Fsymbol isa Number
function Fsymbol(a::I, b::I, c::I, d::I, e::I, f::I) where {I<:S4Irrep}
    (Nsymbol(a, b, e) && Nsymbol(e, c, d) && Nsymbol(b, c, f) && Nsymbol(a, f, d)) ||
        return zero(sectorscalartype(I))
    A = fusiontensor(a, b, e)[:, :, :, 1]
    B = fusiontensor(e, c, d)[:, :, 1, 1]
    C = fusiontensor(b, c, f)[:, :, :, 1]
    D = fusiontensor(a, f, d)[:, :, 1, 1]
    return @tensor conj(D[1, 5]) * conj(C[2, 4, 5]) * A[1, 2, 3] * B[3, 4]
end

# S3 has SimpleFusion -> Rsymbol isa Number
function Rsymbol(a::I, b::I, c::I) where {I<:S4Irrep}
    (Nsymbol(a, b, c) && Nsymbol(b, a, c)) || return zero(sectorscalartype(I))
    A = fusiontensor(a, b, c)[:, :, 1, 1]
    B = fusiontensor(b, a, c)[:, :, 1, 1]
    return @tensor conj(B[1, 2]) * A[2, 1]
end
