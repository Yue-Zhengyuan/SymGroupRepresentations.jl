"""
Construct the projectors to all basis vectors that realizes `irrep`
in a generic representation `rep` (specified by matrices of generators).
```
    Pᵢ  = (d/|G|) ∑_g conj(irrep[g]ᵢᵢ) rep[g]  (i = 1, ..., d)
```
where `d` is the dimension of `irrep`, and `|G|` is the number of elements in the group.

## Arguments
- `elements`: a vector of functions that construct matrices 
    for all elements in the group given the matrices for generators.
"""
function cal_projectors(
    irrep::Vector{M}, rep::Vector{M}, elements::Vector
) where {M<:AbstractMatrix}
    @assert length(irrep) == length(rep)
    # number of group elements
    ng = length(elements)
    # dimension of the irrep
    d = size(irrep[1], 1)
    # TODO: for SNIrrep all possible `irreps` are real,
    # so we assume `rep` is also real. 
    # In principle this condition can be relaxed. 
    projs = Vector{M}(undef, d)
    for i in 1:d
        projs[i] = zeros(eltype(irrep[1]), size(rep[1]))
        for g in elements
            projs[i] += (d / ng) * conj(g(irrep...)[i, i]) * g(rep...)
        end
    end
    return projs
end

"""
    allowed_signs(A, B)

Given two square matrices `A` and `B` related by `A = U * B * U`,
where `U` is a diagonal matrix, with diagonal elements `u[i] = ±1` and `u[1] = +1`,
returns an array of possible `u` vectors (each as a Vector{Int}).

It checks for consistency on the entries where B is nonzero, i.e.,
```
    A[i,j] = u[i] * u[j] * B[i,j]
```
must hold for every `(i,j)` with `B[i,j] ≠ 0`.
"""
function allowed_signs(A::AbstractMatrix, B::AbstractMatrix)
    n, m = size(A)
    if n != m || size(B, 1) != size(B, 2) || n != size(B, 1)
        error("Matrices A and B must be square and of the same size.")
    end
    # Check consistency for zero entries
    for i in 1:n, j in 1:n
        if iszero(B[i, j])
            if A[i, j] != 0
                error("Entry A[$i,$j] must be zero when B[$i,$j] is zero.")
            end
        else
            # When B[i,j] != 0, ratio must be ±1.
            ratio = A[i, j] / B[i, j]
            if !(ratio ≈ 1 || ratio ≈ -1)
                error("For entry ($i,$j), A[i,j]/B[i,j] must be ±1 (got $ratio).")
            end
        end
    end
    # Build the constraint graph
    # For each node i, store a vector of (j, constraint) pairs from A[i,j] and B[i,j].
    graph = [Vector{Tuple{Int,Int}}() for _ in 1:n]
    # We iterate through the entries; in many cases the directed edge is sufficient,
    # but to help propagate constraints we add both directions where possible.
    for i in 1:n
        for j in 1:n
            if !iszero(B[i, j])
                # Compute constraint: u_i * u_j = epsilon  ==> u_j = epsilon * u_i.
                epsilon = Int(round(A[i, j] / B[i, j]; digits=14))
                push!(graph[i], (j, epsilon))
                # It is natural to add the edge from j to i as well (this is safe since if B[j,i] is nonzero
                # it would be added separately anyway, and if B[j,i] is zero, it might not matter; but for propagation
                # in the connected component, we need to know all relationships).
                push!(graph[j], (i, epsilon))
            end
        end
    end
    # Prepare arrays to store the found U values (0 means not yet assigned)
    U_vals = zeros(Int, n)  # will eventually contain ±1
    # For tracking connected components
    comp_id = zeros(Int, n)   # 0 means not assigned, else a positive integer for the component index
    comp_nodes = Dict{Int,Vector{Int}}()
    current_comp = 0
    # We'll propagate the constraints using BFS (or DFS) in each connected component.
    for i in 1:n
        if comp_id[i] == 0
            current_comp += 1
            comp_nodes[current_comp] = Int[]
            # If this is the component containing index 1, fix u₁ = +1.
            if i == 1
                U_vals[i] = 1
            else
                # For new free component, assign an arbitrary value; we can later flip the whole component.
                U_vals[i] = 1
            end
            comp_id[i] = current_comp
            push!(comp_nodes[current_comp], i)
            # Use a queue for BFS
            queue = [i]
            while !isempty(queue)
                cur = popfirst!(queue)
                for (nbr, eps) in graph[cur]
                    # The relation is: u_nbr = eps * u_cur.
                    proposed = eps * U_vals[cur]
                    if comp_id[nbr] == 0
                        comp_id[nbr] = current_comp
                        U_vals[nbr] = proposed
                        push!(comp_nodes[current_comp], nbr)
                        push!(queue, nbr)
                    else
                        # Already assigned; check consistency.
                        if U_vals[nbr] != proposed
                            error(
                                "Inconsistency detected: No U exists that satisfies the transformation.",
                            )
                        end
                    end
                end
            end
        end
    end
    # Now that we have a provisional assignment U_vals, 
    # note that for any component not containing index 1,
    # flipping the sign of all u in that component also gives a valid solution.
    # Find all component IDs that do NOT include index 1.
    anchored_comps = Set{Int}()
    if comp_id[1] != 0
        push!(anchored_comps, comp_id[1])
    end
    free_comps = [comp for comp in keys(comp_nodes) if comp ∉ anchored_comps]
    num_free = length(free_comps)
    # Enumerate over all 2^(num_free) choices for flipping free components.
    results = Vector{Vector{Int}}()
    for s in 0:(2^num_free - 1)
        candidate = copy(U_vals)
        # For each free component, decide whether to flip (if the corresponding bit is 1 then flip).
        for (index, comp) in enumerate(free_comps)
            # Check bit: index-1 bit of s.
            if (s >> (index - 1)) & 1 == 1
                # Flip the sign for every node in this component.
                for node in comp_nodes[comp]
                    candidate[node] = -candidate[node]
                end
            end
        end
        push!(results, candidate)
    end
    return results
end

function _length_to_slice(lengths::Vector{Int})
    start_idx = 1
    slices = map(lengths) do l
        slice = start_idx:(start_idx + l - 1)
        start_idx += l
        return slice
    end
    return slices
end

"""
Find all Clebsch-Gordan coefficients relevant to reduction of `s1 ⊗ s2`.
"""
function cal_CGCs(s1::R, s2::R) where {R<:SNIrrep}
    # s1 ⊗ s2 = s2 ⊗ s1
    if s1 > s2
        return Dict(key[[2, 1, 3, 5, 4, 6]] => val for (key, val) in cal_CGCs(s2, s1))
    end
    if R == S3Irrep
        irrep_gen = irreps_gen.S3
        elements = genreps.S3
    elseif R == S4Irrep
        irrep_gen = irreps_gen.S4
        elements = genreps.S4
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
    c3s = findall([Nsymbol(s1, s2, s3) for s3 in values(R)])
    if isone(s1)
        # trivial case
        basis = Matrix{T}(I(dim(s2)))
    else
        # generator matrix for s1 ⊗ s2
        irrep1, irrep2 = irrep_gen[c1], irrep_gen[c2]
        rep = collect(kron(rep1, rep2) for (rep1, rep2) in zip(irrep1, irrep2))
        # find all projectors
        projs = vcat((cal_projectors(irrep_gen[c3], rep, elements) for c3 in c3s)...)
        basis = hcat(
            (
                map(projs) do p
                    p2 = p - I
                    if (size(p2) == (1, 1)) && (norm(p2, 2) < 5 * eps())
                        p2 .= 0
                    end
                    col = nullspace(p2)
                    if size(col, 2) != 1
                        error(
                            "(Projector - I) null space dimension is not 1 (obtained $(size(col, 2))) for s1 = $s1, s2 = $s2.",
                        )
                    end
                    return col
                end
            )...,
        )
        # adjust sign of columns
        rep2 = [round.(inv(basis) * g * basis; digits=14) for g in rep]
        d3s = [dim(values(R)[c3]) for c3 in c3s]
        slices = _length_to_slice(d3s)
        signs = vcat(
            (
                map(zip(c3s, slices)) do (c3, slice)
                    subrep1 = irrep_gen[c3]
                    subrep2 = [g[slice, slice] for g in rep2]
                    us = intersect(
                        (allowed_signs(g1, g2) for (g1, g2) in zip(subrep1, subrep2))...
                    )
                    return us[1]
                end
            )...,
        )
        basis[:, findall(x -> x == -1, signs)] *= -1
    end
    # meaning of each row/column of `basis`
    rows = [(i1, i2) for i1 in 1:dim(s1) for i2 in 1:dim(s2)]
    cols = [(c3, i3) for c3 in c3s for i3 in 1:dim(values(R)[c3])]
    # convert to CGC dict; entries with CGC = 0 are not saved
    CGC = Dict{NTuple{6,Int},T}()
    for (r, (i1, i2)) in enumerate(rows), (c, (c3, i3)) in enumerate(cols)
        if !isapprox(basis[r, c], 0.0; atol=5 * eps())
            CGC[(c1, c2, c3, i1, i2, i3)] = basis[r, c]
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
