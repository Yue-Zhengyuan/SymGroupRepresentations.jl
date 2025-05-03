# tolerance for nullspace
const TOL_NULLSPACE = 1e-13
# tolerance for gaugefixing should probably be bigger than that with which nullspace was determined
const TOL_GAUGE = 1e-11
# tolerance for dropping zeros
const TOL_PURGE = 1e-14

function qrpos!(C)
    q, r = qr!(C)
    d = diag(r)
    map!(x -> x == zero(x) ? 1 : sign(x), d, d)
    D = Diagonal(d)
    Q = rmul!(Matrix(q), D)
    R = ldiv!(D, Matrix(r))
    return Q, R
end

"""
Find the column reduced echelon form (cref) of matrix `A`
"""
function cref!(
    A::AbstractMatrix,
    ɛ=eltype(A) <: Union{Rational,Integer} ? 0 : 10 * length(A) * eps(norm(A, Inf)),
)
    nr, nc = size(A)
    i = j = 1
    @inbounds while i <= nr && j <= nc
        (m, mj) = findabsmax(view(A, i, j:nc))
        mj = mj + j - 1
        if m <= ɛ
            if ɛ > 0
                A[i, j:nc] .= zero(eltype(A))
            end
            i += 1
        else
            @simd for k in i:nr
                A[k, j], A[k, mj] = A[k, mj], A[k, j]
            end
            d = A[i, j]
            @simd for k in i:nr
                A[k, j] /= d
            end
            for k in 1:nc
                if k != j
                    d = A[i, k]
                    @simd for l in i:nr
                        A[l, k] -= d * A[l, j]
                    end
                end
            end
            i += 1
            j += 1
        end
    end
    return A
end

gaugefix!(C) = first(qrpos!(cref!(C, TOL_GAUGE)))

function findabsmax(a)
    isempty(a) && throw(ArgumentError("collection must be non-empty"))
    m = abs(first(a))
    mi = firstindex(a)
    for (k, v) in pairs(a)
        if abs(v) > m
            m = abs(v)
            mi = k
        end
    end
    return m, mi
end

function _nullspace!(
    A::AbstractMatrix;
    atol::Real=0.0,
    rtol::Real=(min(size(A)...) * eps(real(float(one(eltype(A)))))) * iszero(atol),
)
    m, n = size(A)
    (m == 0 || n == 0) && return Matrix{eltype(A)}(I, n, n)
    SVD = svd!(A; full=true, alg=LinearAlgebra.QRIteration())
    tol = max(atol, SVD.S[1] * rtol)
    indstart = sum(s -> s .> tol, SVD.S) + 1
    return copy(SVD.Vt[indstart:end, :]')
end
