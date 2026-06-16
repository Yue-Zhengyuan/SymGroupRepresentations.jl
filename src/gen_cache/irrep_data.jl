# number of elements in each conjugacy class
const ncs = (; :S3 => [1, 3, 2], :S4 => [1, 6, 3, 8, 6], :S5 => [1, 10, 15, 20, 20, 30, 24])

# character table (ordered by descending partition: trivial → sign)
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
    :S5 => [
        1 1 1 1 1 1 1
        4 2 0 1 -1 0 -1
        5 1 1 -1 1 -1 0
        6 0 -2 0 0 0 1
        5 -1 1 -1 -1 1 0
        4 -2 0 1 1 0 -1
        1 -1 1 1 -1 -1 1
    ],
)

# unitary irrep matrices for generators, computed via Young's orthogonal form.
# Each irrep is [x1, x2] with x1 = (1,2,...,n) and x2 = (1,2).
# The irreps are ordered by descending lexicographic partition (trivial first,
# sign last), matching the character table above.
const irreps_gen = (;
    :S3 => young_orthogonal_irreps(3),
    :S4 => young_orthogonal_irreps(4),
    :S5 => young_orthogonal_irreps(5),
)
