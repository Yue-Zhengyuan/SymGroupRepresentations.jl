"""
    cycle_type_representative(cycle_type, n::Int)

Construct a permutation representative (as a vector `p[1:n]` where `p[i]` is the
image of `i`) whose cycle type is the partition `cycle_type` ⊢ `n`.
"""
function cycle_type_representative(cycle_type, n::Int)
    p = zeros(Int, n)
    pos = 1
    for c in collect(cycle_type)
        for j in 1:(c - 1)
            p[pos + j - 1] = pos + j
        end
        p[pos + c - 1] = pos
        pos += c
    end
    return p
end

"""
    perm_to_adjacent_transpositions(perm::Vector{Int})

Decompose a permutation `perm` (given as a vector where `perm[i]` is the image
of `i`) into a product of adjacent transpositions s_k = (k, k+1).

Returns a vector `ts` of indices such that
`perm = s_{ts[end]} ∘ … ∘ s_{ts[1]}`.
In matrix form: `M_perm = M_{s_{ts[end]}} * … * M_{s_{ts[1]}}`.
"""
function perm_to_adjacent_transpositions(perm::Vector{Int})
    n = length(perm)
    current = copy(perm)
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
    character(irrep::Vector{<:AbstractMatrix}, class_partition)

Compute the character of the representation `irrep = [x1, x2]` on the conjugacy
class with cycle type `class_partition`.  Call this variant to avoid recomputing
the irrep matrices when evaluating many classes against the same representation.
"""
function character(irrep::Vector{<:AbstractMatrix}, class_partition)
    x1, x2 = irrep[1], irrep[2]
    n = sum(class_partition)
    d = size(x1, 1)

    rep = cycle_type_representative(class_partition, n)
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
    class_size(cycle_type)

Compute the number of elements in the conjugacy class of S_n with cycle type
`cycle_type`.

The formula is  n! / (∏ j^{m_j} · m_j!)  where m_j is the multiplicity of
j-cycles in the partition.
"""
function class_size(cycle_type)
    n = sum(cycle_type)
    parts = collect(cycle_type)
    counts = Dict{Int, Int}()
    for c in parts
        counts[c] = get(counts, c, 0) + 1
    end
    denom = prod(c^m * factorial(m) for (c, m) in counts; init = 1)
    return factorial(n) ÷ denom
end
