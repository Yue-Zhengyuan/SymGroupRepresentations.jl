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

    # get irrep matrices for the two generators
    x1, x2 = young_orthogonal_irrep(λ)
    d = size(x1, 1)

    # permutation representative for this cycle type
    rep = cycle_type_representative(μ, n)

    # decompose into adjacent transpositions
    ts = perm_to_adjacent_transpositions(rep)

    if isempty(ts)
        # identity conjugacy class → character is dimension
        return Float64(d)
    end

    # Multiply in reverse: M_rep = M_{s_{t_last}} ⋯ M_{s_{t_first}}
    M = Matrix{Float64}(I, d, d)
    for t in reverse(ts)
        # s_t = x1^{t-1} * x2 * x1^{-(t-1)}
        Ms = x1^(t - 1) * x2 * x1^-(t - 1)
        M = Ms * M
    end

    return real(tr(M))
end

"""
    character_table(n::Int)

Compute the character table of S_n.  Rows correspond to irreducible representations
ordered by descending partition (trivial first, sign last), columns to conjugacy
classes ordered by ascending partition (identity first, n-cycle last).

Returns an `nparts × nparts` integer matrix.
"""
function character_table(n::Int)
    λ_parts = sort!(collect(AbstractAlgebra.Generic.partitions(n)), rev = true)
    μ_parts = sort!(collect(AbstractAlgebra.Generic.partitions(n)))
    ncols = length(μ_parts)

    table = Matrix{Int}(undef, length(λ_parts), ncols)
    for (i, λ) in enumerate(λ_parts)
        x1, x2 = young_orthogonal_irrep(λ)
        d = size(x1, 1)
        for (j, μ) in enumerate(μ_parts)
            table[i, j] = _character_from_irrep(x1, x2, μ, n, d)
        end
    end
    return table
end

"""
    _character_from_irrep(x1, x2, μ, n, d)

Compute a single character value given pre-computed irrep matrices, avoiding
recomputation of `young_orthogonal_irrep`.
"""
function _character_from_irrep(x1, x2, μ, n, d)
    rep = cycle_type_representative(μ, n)
    ts = perm_to_adjacent_transpositions(rep)

    if isempty(ts)
        return d
    end

    M = Matrix{Float64}(I, d, d)
    for t in reverse(ts)
        Ms = x1^(t - 1) * x2 * x1^-(t - 1)
        M = Ms * M
    end

    return round(Int, real(tr(M)))
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
