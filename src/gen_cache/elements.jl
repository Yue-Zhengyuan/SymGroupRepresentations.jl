# elements in the symmetric group expressed in terms of the two generators
# x1 = (1,2,...,n), x2 = (1,2)
#
# Each group element is represented as a callable `GroupElem` storing a word
# (a sequence of generator symbols :x1, :x1inv, :x2) discovered via BFS on the
# permutation group.  Calling it with generator matrices returns the product.

struct GroupElem
    word::Vector{Symbol}
end

function (g::GroupElem)(x1, x2)
    result = x1^0  # identity matrix
    # Right-multiplications in the BFS produce a word [g1, g2, ..., gk] where
    # the final permutation is gk ∘ ... ∘ g1, corresponding to the matrix
    # product M_{gk} * ... * M_{g1}.  We therefore multiply in reverse order.
    for s in reverse(g.word)
        if s == :x1
            result = result * x1
        elseif s == :x1inv
            result = result * inv(x1)
        elseif s == :x2
            result = result * x2
        end
    end
    return result
end

"""
    generate_group_elements(n::Int) -> Vector{GroupElem}

Generate all n! elements of the symmetric group S_n as words in the
standard generators  x1 = (1,2,…,n)  and  x2 = (1,2), using a breadth-first
search on the permutation group.
"""
function generate_group_elements(n::Int)
    # Permutations are represented as vectors p[1:n] where p[i] is the image of i.
    id = collect(1:n)
    # n-cycle (1,2,...,n):     1→2, 2→3, ..., n→1
    x1_perm = circshift(1:n, -1)
    # inverse n-cycle
    x1inv_perm = circshift(1:n, 1)
    # transposition (1,2)
    x2_perm = vcat([2, 1], collect(3:n))

    # Composition: (p ∘ q)(i) = p[q[i]]
    compose(p, q) = [p[q[i]] for i in 1:n]

    # BFS from identity, recording the first word that reaches each permutation
    perm_to_word = Dict(id => Symbol[])
    queue = [id]
    generators = [(:x1, x1_perm), (:x1inv, x1inv_perm), (:x2, x2_perm)]

    N = factorial(n)
    while !isempty(queue) && length(perm_to_word) < N
        p = popfirst!(queue)
        word = perm_to_word[p]
        for (sym, gen_perm) in generators
            new_perm = compose(gen_perm, p)   # right-multiplication: gen ∘ p
            if !haskey(perm_to_word, new_perm)
                perm_to_word[new_perm] = vcat(word, [sym])
                push!(queue, new_perm)
            end
        end
    end

    @assert length(perm_to_word) == N "Expected $N elements, found $(length(perm_to_word))"

    return [GroupElem(word) for (_, word) in perm_to_word]
end

genreps = (;
    :S3 => generate_group_elements(3),
    :S4 => generate_group_elements(4),
    :S5 => generate_group_elements(5),
)
