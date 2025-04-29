"""
Given a non-unitary representation `rep` of a group with `elements`,
convert it to a unitary representation using the "unitarian trick"
(See Chapter II.1 in Zee, Group Theory in A Nutshell for Physicists).

## Arguments
- `rep::Vector{T} where {T<:AbstractMatrix}`: the matrix for group generators in the non-unitary representation.
- `elements::Vector`: functions to calculate matrices for all group elements from the generators.
"""
function get_unitary_rep(rep::Vector{T}, elements::Vector) where {T<:AbstractMatrix}
    if all(is_unitary.(rep))
        return rep
    end
    ng = length(elements)
    H = (1 / ng) * sum(
        map(elements) do elem
            g = elem(rep...)
            return g' * g
        end,
    )
    F = eigen(H)
    @assert all(e .>= 0 for e in F.values)
    ρ = diagm(sqrt.(F.values))
    ρinv = diagm(1 ./ sqrt.(F.values))
    W = F.vectors
    rep_u = [ρ * W' * g * W * ρinv for g in rep]
    # self-check
    @assert all(is_unitary.(rep_u))
    for (g, gu) in zip(rep, rep_u)
        if isapprox(tr(g), 0; atol=1e-14)
            @assert isapprox(tr(gu), 0, atol=1e-14)
        else
            @assert isapprox(tr(g), tr(gu))
        end
    end
    return rep_u
end

"""
Determine if a square matrix is unitary
"""
function is_unitary(a::AbstractMatrix)
    @assert size(a, 1) == size(a, 2)
    n = size(a, 1)
    iden = Matrix{eltype(a)}(I, n, n)
    return isapprox(a' * a, iden; atol=1e-12)
end
