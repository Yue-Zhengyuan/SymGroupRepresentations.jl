# number of elements in each conjugacy class
const ncs = (;
    :S3 => [class_size(μ) for μ in sort!(partitions(3); rev = true)],
    :S4 => [class_size(μ) for μ in sort!(partitions(4); rev = true)],
    :S5 => [class_size(μ) for μ in sort!(partitions(5); rev = true)],
)

# character table (rows: descending partition order, cols: ascending partition order)
const char_tables = (;
    :S3 => character_table(3),
    :S4 => character_table(4),
    :S5 => character_table(5),
)

# unitary irrep matrices for generators, computed via Young's orthogonal form.
# Each irrep is [x1, x2] with x1 = (1,2,...,n) and x2 = (1,2).
const irreps_gen = (;
    :S3 => young_orthogonal_irreps(3),
    :S4 => young_orthogonal_irreps(4),
    :S5 => young_orthogonal_irreps(5),
)
