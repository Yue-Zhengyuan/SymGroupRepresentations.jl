function _length_to_slice(lengths::Vector{Int}, muls::Vector{Int})
    slices = Vector{UnitRange{Int}}()
    start = 1
    for (len, count) in zip(lengths, muls)
        for _ in 1:count
            slice = start:(start + len - 1)
            push!(slices, slice)
            start += len
        end
    end
    return slices
end

"""
Find all Clebsch-Gordan coefficients relevant to reduction of `s1 ⊗ s2`.
"""
function cal_CGCs(s1::R, s2::R) where {R<:SNIrrep}
    # s1 ⊗ s2 = s2 ⊗ s1
    if s1 > s2
        return Dict(key[[2, 1, 3, 5, 4, 6, 7]] => val for (key, val) in cal_CGCs(s2, s1))
    end
    if R == S3Irrep
        irrep_gen = irreps_gen.S3
        elements = genreps.S3
    elseif R == S4Irrep
        irrep_gen = irreps_gen.S4
        elements = genreps.S4
    elseif R == S5Irrep
        irrep_gen = irreps_gen.S5
        elements = genreps.S5
    else
        error("$R is not implemented.")
    end
    c1, c2 = 0, 0
    for (a, s) in enumerate(values(R))
        (s == s1) && (c1 = a)
        (s == s2) && (c2 = a)
        (c1 != 0) && (c2 != 0) && break
    end
    T = eltype(irrep_gen[c2][1])
    # find irreps appearing in s1 ⊗ s2
    n3s = [Int(Nsymbol(s1, s2, s3)) for s3 in values(R)]
    c3s = findall(x -> x > 0, n3s)
    n3s = n3s[c3s]
    if isone(s1)
        # trivial case
        cgbasis = Matrix{T}(I(dim(s2)))
    else
        # generator matrix for s1 ⊗ s2
        irrep1, irrep2 = irrep_gen[c1], irrep_gen[c2]
        rep = collect(kron(rep1, rep2) for (rep1, rep2) in zip(irrep1, irrep2))
        cgbasis = hcat((
            map(zip(c3s, n3s)) do (c3, n3)
                irrep3 = irrep_gen[c3]
                if n3 == 1
                    basis = get_intertwiner(irrep3, rep, elements)
                    num = _find_first_nonzero_element(basis)
                    basis .*= abs(num) / num
                    @assert is_left_unitary(basis)
                else
                    error("Not implemented")
                end
                return basis
            end
        )...)
    end
    # meaning of each row/column of `basis`
    rows = [(i1, i2) for i1 in 1:dim(s1) for i2 in 1:dim(s2)]
    cols = [
        (c3, i3, deg) for (a, c3) in enumerate(c3s) for i3 in 1:dim(values(R)[c3]) for
        deg in 1:n3s[a]
    ]
    # convert to CGC dict; entries with CGC = 0 are not saved
    CGC = Dict{NTuple{7,Int},T}()
    for (r, (i1, i2)) in enumerate(rows), (c, (c3, i3, deg)) in enumerate(cols)
        if !isapprox(cgbasis[r, c], 0.0; atol=5 * eps())
            CGC[(c1, c2, c3, i1, i2, i3, deg)] = cgbasis[r, c]
        end
    end
    return CGC
end

"""
Find all Clebsch-Gordan coefficients for `SNIrrep`
"""
function calall_CGCs(R::Type{<:SNIrrep})
    return merge((cal_CGCs(s1, s2) for s1 in values(R), s2 in values(R))...)
end
