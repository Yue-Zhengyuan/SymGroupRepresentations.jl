"""
Determine if a square matrix `a` is left-unitary, i.e. `a' * a ≈ 1`.
"""
function is_left_unitary(a::AbstractMatrix; tol=1e-12)
    n = size(a, 2)
    iden = Matrix{eltype(a)}(I, n, n)
    return isapprox(a' * a, iden; atol=tol)
end

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

"""
Given a non-unitary representation `rep` of a group with `elements`,
convert it to a unitary representation using the "unitarian trick"
(See Chapter II.1 in Zee, Group Theory in A Nutshell for Physicists).

## Arguments
- `rep::Vector{T} where {T<:AbstractMatrix}`: the matrix for group generators in the non-unitary representation.
- `elements::Vector`: functions to calculate matrices for all group elements from the generators.
"""
function get_unitary_rep(rep::Vector{T}, elements::Vector) where {T<:AbstractMatrix}
    if all(is_left_unitary.(rep))
        return rep
    end
    ng = length(elements)
    H = similar(rep[1])
    fill!(H, 0)
    for elem in elements
        g = elem(rep...)
        H += (1 / ng) * (g' * g)
    end
    F = eigen(H)
    @assert all(e .>= 0 for e in F.values)
    ρ = diagm(sqrt.(F.values))
    ρinv = diagm(1 ./ sqrt.(F.values))
    U = F.vectors * ρinv
    Uinv = ρ * (F.vectors)'
    rep_u = [Uinv * g * U for g in rep]
    # self-check
    @assert all(is_left_unitary.(rep_u))
    for (g, gu) in zip(rep, rep_u)
        if isapprox(tr(g), 0; atol=1e-14)
            @assert isapprox(tr(gu), 0, atol=1e-14)
        else
            @assert isapprox(tr(g), tr(gu))
        end
    end
    return rep_u
end

function _find_first_nonzero_element(matrix::AbstractMatrix, tol=1e-14)
    for a in matrix[:, 1]
        (abs(a) > tol) && return a
    end
    return error("No element in the first column is significantly different from zero.")
end

function is_propto1(A::AbstractMatrix, tol=1e-12)
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
(`rep1` is irreducible, and `rep2` has a larger or equal dimension than `rep1`)
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
function get_intertwiners(rep1::Vector{M}, rep2::Vector{M}) where {M<:AbstractMatrix}
    d1, d2 = size(rep1[1], 1), size(rep2[1], 1)
    (d2 < d1) && error("Dimension of `rep2` must be larger than `rep1`.")
    L = vcat(
        (kron(I(d1), r2) - kron(transpose(r1), I(d2)) for (r1, r2) in zip(rep1, rep2))...
    )
    # intertwiner space is the same as null space of `op`
    # fs = nullspace(L)
    # fs = [polar(Matrix(reshape(f, (d2, d1)))).U for f in eachcol(fs)]
    fs = _nullspace!(L; atol=TOL_NULLSPACE)
    fs = [gaugefix!(Matrix(reshape(f, (d2, d1)))) for f in eachcol(fs)]
    (length(fs) == 0) &&
        error("There are no non-trivial intertwiners between rep1 and rep2.")
    return fs
end
