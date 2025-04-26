# number of elements in each conjugacy class
const ncs = (; :S3 => [1, 3, 2], :S4 => [1, 6, 3, 8, 6])

# character table
const char_tables = (;
    :S3 => [
        1 1 1
        2 0 -1
        1 -1 1
    ],
    :S4 => [
        1 1 1 1 1
        3 1 -1 0 -1
        2 0 2 -1 0
        3 -1 -1 0 1
        1 -1 1 1 -1
    ],
)

# unitary irrep matrices for generators
const irreps_gen = (;
    :S3 => [
        [[1;;], [1;;]],
        [[1/2 -sqrt(3)/2; -sqrt(3)/2 -1/2], [-1 0; 0 1]],
        [[-1;;], [-1;;]],
    ],
    :S4 => [
        [[1;;], [1;;]],
        [[-1 0 0; 0 0 1; 0 -1 0], [0 -1 0; -1 0 0; 0 0 1]],
        [[1/2 -sqrt(3)/2; -sqrt(3)/2 -1/2], [-1 0; 0 1]],
        [[1 0 0; 0 0 -1; 0 1 0], [0 1 0; 1 0 0; 0 0 -1]],
        [[-1;;], [-1;;]],
    ],
)

# elements in the group expressed in terms of the two generators
# x1 = (1,2,...,n), x2 = (1,2)
genreps = (;
    :S3 =>
        ((x1, x2) -> [Matrix{Float64}(I(size(x1, 1))), x1 * x2, x2, x1, inv(x1), x2 * x1]),
    :S4 => (
        (x1, x2) -> [
            Matrix{Float64}(I(size(x1, 1))),
            x1^2 * x2 * x1^2,
            inv(x1) * x2 * x1,
            x1 * x2,
            x2 * inv(x1),
            x2 * x1^2 * x2 * x1,
            x2,
            (x2 * x1^2)^2,
            inv(x1) * x2 * x1 * x2,
            x1,
            x1 * x2 * x1,
            inv(x1) * x2 * x1^2,
            x1 * x2 * x1^2,
            x2 * x1 * x2,
            x1 * x2 * x1^2 * x2,
            x2 * x1,
            x1^2,
            x1^2 * x2,
            inv(x1),
            x1^2 * x2 * x1,
            inv(x1) * x2,
            x1 * x2 * inv(x1),
            x2 * x1^2,
            x2 * x1^2 * x2,
        ]
    ),
)