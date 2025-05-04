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
Since `s1 ⊗ s2 = s2 ⊗ s1`, we only calculate `s1 ≤ s2` cases. 
"""
function _cal_CGCs(s1::R, s2::R) where {R<:SNIrrep}
    (s1 > s2) && error("Only intended to calculate CGCs for `s1 ≤ s2`.")
    if R == S3Irrep
        irrep_gen = irreps_gen.S3
    elseif R == S4Irrep
        irrep_gen = irreps_gen.S4
    elseif R == S5Irrep
        irrep_gen = irreps_gen.S5
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
        cgbasis = hcat(
            (
                map(zip(c3s, n3s)) do (c3, n3)
                    irrep3 = irrep_gen[c3]
                    fs = get_intertwiners(irrep3, rep)
                    iden = collect(_inner_prod(f1, f2) for f1 in fs, f2 in fs)
                    @assert isapprox(iden, I(length(fs))) "Intertwiner basis is not orthonormal for s1 = $s1, s2 = $s2, s3 = $(values(R)[c3]). \n$iden"
                    basis = hcat(fs...)
                    @assert is_left_unitary(basis)
                    return basis
                end
            )...,
        )
        # self check: compare with the direct sum representation
        @assert is_left_unitary(cgbasis) "CG basis is not unitary for $s1 ⊗ $s2."
        rep = [cgbasis' * g * cgbasis for g in rep]
        repds = [
            block_diag(
                (irrep_gen[c3][i] for (a, c3) in enumerate(c3s) for _ in 1:n3s[a])...
            ) for i in 1:length(rep)
        ]
        if !all(isapprox(g1, g2) for (g1, g2) in zip(rep, repds))
            error("Calculated CG basis is incorrect for irrep decomposition of $s1 ⊗ $s2.")
        end
    end
    # meaning of each row/column of `basis`
    rows = [(i1, i2) for i1 in 1:dim(s1) for i2 in 1:dim(s2)]
    cols = [
        (c3, i3, deg) for (a, c3) in enumerate(c3s) for deg in 1:n3s[a] for
        i3 in 1:dim(values(R)[c3])
    ]
    # convert to CGC dict; entries with CGC = 0 are not saved
    CGC = Dict{NTuple{7,Int},T}()
    for (r, (i1, i2)) in enumerate(rows), (c, (c3, i3, deg)) in enumerate(cols)
        if abs(cgbasis[r, c]) > TOL_PURGE
            CGC[(c1, c2, c3, i1, i2, i3, deg)] = cgbasis[r, c]
        end
    end
    return CGC
end

"""
Find all nonzero Clebsch-Gordan coefficients for `SNIrrep` with `s1 ≤ s2`.
"""
function _calall_CGCs(R::Type{<:SNIrrep})
    # CGCs with s1 <= s2
    ss = values(R)
    n = length(ss)
    allcgcs = merge((begin
        s1, s2 = ss[c1], ss[c2]
        _cal_CGCs(s1, s2)
    end for c1 in 1:n for c2 in c1:n)...)
    return allcgcs
end
