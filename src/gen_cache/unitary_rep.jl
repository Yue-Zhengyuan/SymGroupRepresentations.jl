"""
Create a block diagonal matrix from square blocks
"""
function block_diag(matrices::AbstractMatrix...)
    n = sum(size(matrix, 1) for matrix in matrices)
    bm = spzeros(n, n)
    r = 1
    for matrix in matrices
        rows, cols = size(matrix)
        @assert rows == cols
        bm[r:(r + rows - 1), r:(r + cols - 1)] = matrix
        r += rows
    end
    return bm
end

function is_propto1(A::AbstractMatrix, tol = 1.0e-12)
    if (size(A, 1) == size(A, 2))
        n = size(A, 1)
        c1 = (tr(A) / n) * Matrix{eltype(A)}(I(n))
        return norm(A - c1) < tol
    else
        return false
    end
end

"""
Inner product of two intertwiners `f1, f2` between two unitary representations
`rep1`, `rep2`, where `rep2` is irreducible and `rep1` has a larger dimension.
"""
function _inner_prod(f1::AbstractMatrix, f2::AbstractMatrix)
    @assert size(f1) == size(f2)
    @assert size(f1, 1) >= size(f1, 2)
    mat = f1' * f2
    @assert is_propto1(mat)
    return mat[1, 1]
end

"""
    get_intertwiners(rep1::Vector{M}, rep2::Vector{M}) where {M<:AbstractMatrix}

Given two unitary representations `rep1`, `rep2` of a group `G`
(assume `rep1` is irreducible)
construct an orthonormal basis of the space of intertwiners `f`
between the two representations that satisfies
```
    ∀ g ∈ G:    rep2[g] ∘ f = f ∘ rep1[g]
```
The basis intertwiners are orthonormal in the sense that
```
    fᵢ' * fⱼ = δᵢⱼ 1
```
where `1` is the identity matrix with the same dimension as `rep1`. 

## Arguments

- `rep1::Vector{M}`: matrices of group generators in `rep1`.
- `rep2::Vector{M}`: matrices of group generators in `rep2` (with the same `eltype` as `rep1`).

## Details
The intertwiner equation can be rewritten as
```
    L[g] vec(f) = 0,  where  L[g] = 1 ⊗ rep2[g] - transpose(rep1[g]) ⊗ 1
```
Thus the space of intertwiners is just the common null space (kernel) of each `L[g]`.
Actually, using `L[g]` for group generators is enough. 
"""
function get_intertwiners(rep1::Vector{M}, rep2::Vector{M}) where {M <: AbstractMatrix}
    d1, d2 = size(rep1[1], 1), size(rep2[1], 1)
    # (d2 < d1) && error("Dimension of `rep2` must be larger than `rep1`.")
    L = vcat(
        (kron(I(d1), r2) - kron(transpose(r1), I(d2)) for (r1, r2) in zip(rep1, rep2))...
    )
    fs = nullspace(L; atol = TOL_NULLSPACE)
    # make the null space basis vectors unique
    fs = gaugefix!(fs)
    @assert isisometric(fs)
    fs = [first(left_polar(Matrix(reshape(f, (d2, d1))))) for f in eachcol(fs)]
    return fs
end
