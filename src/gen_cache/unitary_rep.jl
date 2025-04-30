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

"""
    get_intertwiner(
        rep1::Vector{M}, rep2::Vector{M}, elements::Vector, [S::M]
    ) where {M<:AbstractMatrix}

Given two unitary representations `rep1`, `rep2` of a group `G` 
containing elements `elements`, and a matrix `S`,
(these matrices should have the same `eltype`)
construct an intertwiner `f` between `rep1`, `rep2` as
```
    f = (1/|G|) ∑_g rep2[g] * S * rep1†[g]
```
such that
```
    ∀ g ∈ G:    rep2[g] ∘ f = f ∘ rep1[g]
```
For convenience, we require dimension of `rep1` to be smaller than `rep2`.
"""
function get_intertwiner(
    rep1::Vector{M}, rep2::Vector{M}, elements::Vector, S::M
) where {M<:AbstractMatrix}
    ng = length(elements)
    @assert length(rep1) == length(rep2)
    # check unitarity
    @assert all(is_left_unitary.(rep1))
    @assert all(is_left_unitary.(rep2))
    # Dimension of the representation
    d1, d2 = size(rep1[1], 1), size(rep2[1], 1)
    @assert d1 <= d2
    T = eltype(rep1[1])
    f = zeros(T, d2, d1)
    for g in elements
        f += (1 / ng) * g(rep2...) * S * g(rep1...)'
    end
    return f
end

function get_intertwiner(
    rep1::Vector{M}, rep2::Vector{M}, elements::Vector
) where {M<:AbstractMatrix}
    d1, d2 = size(rep1[1], 1), size(rep2[1], 1)
    T = eltype(rep1[1])
    # avoid the trivial intertwiner
    f = nothing
    while true
        S = rand(T, d2, d1)
        f = get_intertwiner(rep1, rep2, elements, S)
        (norm(f) > 1e-12) && break
    end
    # normalize intertwiner
    λ = (f' * f)[1, 1]
    f ./= sqrt(λ)
    return f
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
Check linear independence of a set of intertwiners.
Returns the check result and the Gram matrix. 
"""
function is_linearly_independent(
    vecs::Vector{M}, tol::Float64=1e-10
) where {M<:AbstractMatrix}
    G = collect(_inner_prod(v1, v2) for v1 in vecs, v2 in vecs)
    return abs(det(G)) > tol, G
end

"""
Gram-Schmidt orthogonalization of a basis of the intertwiner space.
"""
function _gram_schmidt(
    intertwiners::Vector{M}; tol::Float64=1e-10
) where {M<:AbstractMatrix}
    N = length(intertwiners)
    T = eltype(intertwiners[1])
    basis = Vector{M}()
    for i in 1:N
        v = intertwiners[i]
        # Subtract projections along all previously obtained basis vectors.
        for Q in basis
            proj_coeff = _inner_prod(Q, v) / _inner_prod(Q, Q)
            v = v - proj_coeff * Q
        end
        if abs(_inner_prod(v, v)) < tol
            error("Encountered near-zero vector during orthogonalization.")
        end
        norm_v = sqrt(_inner_prod(v, v))
        push!(basis, v / norm_v)
    end
    return basis
end
