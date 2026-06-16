"""
    cycle_type_representative(μ, n)

Construct a permutation representative (as a vector `p[1:n]` where `p[i]` is the
image of `i`) whose cycle type is the partition `μ` ⊢ `n`.
"""
function cycle_type_representative(μ, n::Int)
    p = zeros(Int, n)
    pos = 1
    for c in collect(μ)
        for j in 1:(c - 1)
            p[pos + j - 1] = pos + j
        end
        p[pos + c - 1] = pos
        pos += c
    end
    return p
end

"""
    perm_to_adjacent_transpositions(p)

Decompose a permutation `p` (given as a vector where `p[i]` is the image of `i`)
into a product of adjacent transpositions s_k = (k, k+1).

Returns a vector `ts` of indices such that `p = s_{ts[end]} ∘ … ∘ s_{ts[1]}`.
In matrix form: `M_p = M_{s_{ts[end]}} * … * M_{s_{ts[1]}}`.
"""
function perm_to_adjacent_transpositions(p)
    n = length(p)
    current = copy(p)
    ts = Int[]

    for target in n:-1:2
        idx = findfirst(x -> x == target, current)
        while idx < target
            current[idx], current[idx + 1] = current[idx + 1], current[idx]
            push!(ts, idx)
            idx += 1
        end
    end
    return ts
end

"""
    character(λ, μ)

Compute the character χ_λ(C_μ) — the trace of the irreducible representation of S_n
labelled by partition `λ` on the conjugacy class with cycle type `μ`.

Both `λ` and `μ` are partitions of the same `n`.

# Examples

```julia-repl
julia> character(Partition([3]), Partition([3]))
1.0

julia> character(Partition([2,1]), Partition([2,1]))
1.0
```
"""
function character(λ, μ)
    n = sum(λ)
    @assert sum(μ) == n "Partitions must have the same sum, got $(sum(λ)) and $(sum(μ))"
    return character(young_orthogonal_irrep(λ), μ)
end

"""
    character(irrep::Vector{<:AbstractMatrix}, μ)

Compute the character of the representation `irrep = [x1, x2]` on the conjugacy
class with cycle type `μ`.  Call this variant to avoid recomputing the irrep
matrices when evaluating many classes against the same representation.
"""
function character(irrep::Vector{<:AbstractMatrix}, μ)
    x1, x2 = irrep[1], irrep[2]
    n = sum(μ)
    d = size(x1, 1)

    rep = cycle_type_representative(μ, n)
    ts = perm_to_adjacent_transpositions(rep)

    if isempty(ts)
        return Float64(d)
    end

    M = Matrix{Float64}(I, d, d)
    for t in reverse(ts)
        Ms = x1^(t - 1) * x2 * x1^-(t - 1)
        M = Ms * M
    end

    return real(tr(M))
end

"""
    character_table(n::Int)

Compute the character table of S_n.  Rows correspond to irreducible representations
and columns to conjugacy classes, both ordered by descending partition.

Returns an `nparts × nparts` integer matrix.
"""
function character_table(n::Int)
    λ_parts = sort!(partitions(n); rev = true)
    μ_parts = sort!(partitions(n); rev = true)
    ncols = length(μ_parts)
    table = Matrix{Int}(undef, length(λ_parts), ncols)
    for (i, λ) in enumerate(λ_parts)
        irrep = young_orthogonal_irrep(λ)
        for (j, μ) in enumerate(μ_parts)
            table[i, j] = round(Int, character(irrep, μ))
        end
    end
    return table
end

"""
    class_size(μ)

Compute the number of elements in the conjugacy class of S_n with cycle type `μ`.

The formula is  n! / (∏ j^{m_j} · m_j!)  where m_j is the multiplicity of
j-cycles in μ.
"""
function class_size(μ)
    n = sum(μ)
    parts = collect(μ)
    counts = Dict{Int, Int}()
    for c in parts
        counts[c] = get(counts, c, 0) + 1
    end
    denom = prod(c^m * factorial(m) for (c, m) in counts; init = 1)
    return factorial(n) ÷ denom
end
