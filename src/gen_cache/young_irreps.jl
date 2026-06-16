"""
    all_standard_tableaux(partition)

Return a vector of all standard Young tableaux of shape `partition`, each
represented as a dense `Matrix{Int}` (zeros for empty cells).
"""
function all_standard_tableaux(partition)
    part = collect(partition)
    n = sum(part)
    nrows = length(part)
    ncols = maximum(part)

    tableaux = Matrix{Int}[]

    # current shape (filled cells per row)
    shape = zeros(Int, nrows)
    # tableau under construction
    T = zeros(Int, nrows, ncols)

    function dfs(k)
        if k > n
            push!(tableaux, copy(T))
            return
        end
        # try each addable row
        for r in 1:nrows
            c = shape[r] + 1
            c <= part[r] || continue   # row full
            # Young-diagram condition: cell above must already exist
            (r == 1 || shape[r - 1] >= c) || continue
            T[r, c] = k
            shape[r] += 1
            dfs(k + 1)
            shape[r] -= 1
            T[r, c] = 0
        end
        return
    end

    dfs(1)
    return tableaux
end

"""
    young_orthogonal_irrep(partition)

Return `[x1, x2]` — the matrices of the generators `x1 = (1,2,…,n)` and
`x2 = (1,2)` in Young's orthogonal form for the irrep labelled by `partition`.
The matrices are real orthogonal (hence unitary).
"""
function young_orthogonal_irrep(partition)
    part = collect(partition)
    n = sum(part)
    nrows = length(part)
    ncols = maximum(part)

    # --- enumerate standard Young tableaux ---
    tableaux = all_standard_tableaux(partition)
    d = length(tableaux)

    # --- position lookup: pos[t][v] = (row, col) of value v in tableau t ---
    pos = [zeros(Int, n, 2) for _ in 1:d]
    for (t, T) in enumerate(tableaux)
        for r in 1:nrows, c in 1:part[min(r, end)]  # only iterate valid cells
            val = T[r, c]
            if val > 0
                pos[t][val, 1] = r
                pos[t][val, 2] = c
            end
        end
    end

    # --- tableau → index lookup (via string key) ---
    t_idx = Dict{String, Int}()
    for (i, T) in enumerate(tableaux)
        # Build a canonical string key
        key = join([join(T[r, 1:part[min(r, end)]], ",") for r in 1:nrows], ";")
        t_idx[key] = i
    end

    # --- build s_k matrices ---
    s = [zeros(d, d) for _ in 1:(n - 1)]

    for i in 1:d  # for each tableau
        p = pos[i]

        for k in 1:(n - 1)
            r1, c1 = p[k, 1], p[k, 2]
            r2, c2 = p[k + 1, 1], p[k + 1, 2]

            # axial distance from k+1 to k
            ax = (c2 - c1) - (r2 - r1)

            T = tableaux[i]

            if ax == 1
                # same row, adjacent columns → +1 on diagonal
                s[k][i, i] = 1.0
            elseif ax == -1
                # same column, adjacent rows → −1 on diagonal
                s[k][i, i] = -1.0
            else
                # swap k ↔ k+1 to get s_k·T
                T_swapped = copy(T)
                T_swapped[r1, c1] = k + 1
                T_swapped[r2, c2] = k

                key = join([join(T_swapped[rr, 1:part[min(rr, end)]], ",") for rr in 1:nrows], ";")
                j = t_idx[key]

                s[k][i, i] = 1.0 / ax
                s[k][i, j] = sqrt(1.0 - 1.0 / ax^2)
                # s[k][j,j] = −1/ax  and  s[k][j,i] = s[k][i,j]
                # will be set when the loop reaches tableau j
            end
        end
    end

    # Symmetrise to clean up floating-point noise
    for k in 1:(n - 1)
        sk = s[k]
        s[k] = 0.5 * (sk + sk')
    end

    # --- assemble generator matrices ---
    x2 = s[1]        # (1,2) = s₁
    x1 = Matrix{Float64}(I, d, d)
    for k in 1:(n - 1)
        x1 = x1 * s[k]
    end

    return [x1, x2]
end

"""
    young_orthogonal_irreps(n::Int)

Return a vector of `[x1, x2]` generator pairs for all irreducible representations
of S_n, ordered by descending partition (trivial first, sign last).

# Details

For a partition λ ⊢ n, the irrep has dimension d = number of standard Young
tableaux of shape λ.  The basis vectors are indexed by standard Young tableaux.

The action of an adjacent transposition s_k = (k, k+1) on a tableau T is
governed by the *axial distance* r from k+1 to k in T:

    r = (col_{k+1} − col_k) − (row_{k+1} − row_k)

  • r = +1 (same row, adjacent columns):  s_k · v_T = v_T
  • r = −1 (same column, adjacent rows):  s_k · v_T = −v_T
  • |r| > 1 (different rows & columns):
        s_k · v_T   = (1/r) v_T   + √(1 − 1/r²) v_{s_k T}
        s_k · v_{s_k T} = √(1 − 1/r²) v_T   − (1/r) v_{s_k T}

These matrices are real orthogonal (hence unitary).

The standard generators of S_n used in this package:
  x1 = (1,2,…,n)  =  s₁·s₂·…·s_{n−1}
  x2 = (1,2)      =  s₁

Reference: James & Kerber, "The Representation Theory of the Symmetric Group"
"""
function young_orthogonal_irreps(n::Int)
    parts = sort!(partitions(n); rev = true)
    return [young_orthogonal_irrep(p) for p in parts]
end
