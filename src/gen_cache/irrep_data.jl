# number of elements in each conjugacy class (ascending partition order)
const ncs = (;
    :S3 => [class_size(μ) for μ in sort!(collect(partitions(3)))],
    :S4 => [class_size(μ) for μ in sort!(collect(partitions(4)))],
    :S5 => [class_size(μ) for μ in sort!(collect(partitions(5)))],
)

# character table (rows: descending partition order, cols: ascending partition order)
const char_tables = (;
    :S3 => character_table(3),
    :S4 => character_table(4),
    :S5 => character_table(5),
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
