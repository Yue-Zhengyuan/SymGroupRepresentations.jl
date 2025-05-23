# elements in the symmetric group expressed in terms of the two generators
# x1 = (1,2,...,n), x2 = (1,2)
genreps = (;
    :S3 => [
        (x1, x2) -> x1^0,
        (x1, x2) -> x1 * x2,
        (x1, x2) -> x2,
        (x1, x2) -> x1,
        (x1, x2) -> x1^-1,
        (x1, x2) -> x2 * x1,
    ],
    :S4 => [
        (x1, x2) -> x1^0,
        (x1, x2) -> x1^2 * x2 * x1^2,
        (x1, x2) -> x1^-1 * x2 * x1,
        (x1, x2) -> x1 * x2,
        (x1, x2) -> x2 * x1^-1,
        (x1, x2) -> x2 * x1^2 * x2 * x1,
        (x1, x2) -> x2,
        (x1, x2) -> (x2 * x1^2)^2,
        (x1, x2) -> x1^-1 * x2 * x1 * x2,
        (x1, x2) -> x1,
        (x1, x2) -> x1 * x2 * x1,
        (x1, x2) -> x1^-1 * x2 * x1^2,
        (x1, x2) -> x1 * x2 * x1^2,
        (x1, x2) -> x2 * x1 * x2,
        (x1, x2) -> x1 * x2 * x1^2 * x2,
        (x1, x2) -> x2 * x1,
        (x1, x2) -> x1^2,
        (x1, x2) -> x1^2 * x2,
        (x1, x2) -> x1^-1,
        (x1, x2) -> x1^2 * x2 * x1,
        (x1, x2) -> x1^-1 * x2,
        (x1, x2) -> x1 * x2 * x1^-1,
        (x1, x2) -> x2 * x1^2,
        (x1, x2) -> x2 * x1^2 * x2,
    ],
    :S5 => [
        (x1, x2) -> x1^0,
        (x1, x2) -> x1^2 * x2 * x1^-2,
        (x1, x2) -> x1^-2 * x2 * x1^2,
        (x1, x2) -> x1 * x2 * x1^-1 * x2 * x1,
        (x1, x2) -> x1^-1 * x2 * x1 * x2 * x1^-1,
        (x1, x2) -> x1^-1 * x2 * (x1 * x2 * x1)^2 * x1,
        (x1, x2) -> x1^-1 * x2 * x1,
        (x1, x2) -> (x1^2 * x2)^2 * x1,
        (x1, x2) -> x1^2 * x2 * x1^-1 * x2,
        (x1, x2) -> x1 * x2,
        (x1, x2) -> x1 * x2 * x1^-1 * x2 * x1^2 * x2,
        (x1, x2) -> (x1^-2 * x2)^2,
        (x1, x2) -> x2 * x1 * x2 * x1^-2,
        (x1, x2) -> x1 * x2 * x1^2 * x2 * x1^-1 * x2,
        (x1, x2) -> x2 * (x1 * x2 * x1)^2,
        (x1, x2) -> x1^-1 * x2 * x1^2 * x2,
        (x1, x2) -> (x1 * x2)^2,
        (x1, x2) -> (x1 * x2)^2 * x1^-1 * x2 * x1,
        (x1, x2) -> x2 * x1^-1,
        (x1, x2) -> (x2 * x1^2)^2,
        (x1, x2) -> x2 * x1^-2 * x2 * x1,
        (x1, x2) -> x2 * x1 * x2 * x1^-1 * x2,
        (x1, x2) -> x1^-1 * x2 * x1 * (x1 * x2)^2,
        (x1, x2) -> x2 * x1^-1 * x2 * (x1 * x2 * x1)^2,
        (x1, x2) -> x2,
        (x1, x2) -> x1^2 * x2 * x1^-2 * x2,
        (x1, x2) -> x2 * x1^-2 * x2 * x1^2,
        (x1, x2) -> x2 * x1 * x2 * x1^-1 * x2 * x1,
        (x1, x2) -> x1^-1 * x2 * x1 * x2 * x1^-1 * x2,
        (x1, x2) -> x2 * x1^-1 * x2 * (x1 * x2 * x1)^2 * x1,
        (x1, x2) -> x1^-1 * x2 * x1 * x2,
        (x1, x2) -> (x1^2 * x2)^2 * x1 * x2,
        (x1, x2) -> x1^2 * x2 * x1^-1,
        (x1, x2) -> x1,
        (x1, x2) -> x1 * x2 * x1^-1 * x2 * x1^2,
        (x1, x2) -> x1^-2 * x2 * x1^-2,
        (x1, x2) -> x2 * x1 * x2 * x1^-2 * x2,
        (x1, x2) -> x1 * x2 * x1^2 * x2 * x1^-1,
        (x1, x2) -> (x1^-1 * x2 * x1^-1)^2,
        (x1, x2) -> x1^-1 * x2 * x1^2,
        (x1, x2) -> x1 * x2 * x1,
        (x1, x2) -> x1^2 * x2 * x1^-1 * x2 * x1,
        (x1, x2) -> x2 * x1^-1 * x2,
        (x1, x2) -> (x2 * x1^2)^2 * x2,
        (x1, x2) -> x2 * x1^-2 * x2 * x1 * x2,
        (x1, x2) -> x2 * x1 * x2 * x1^-1,
        (x1, x2) -> x1^-1 * x2 * x1^2 * x2 * x1,
        (x1, x2) -> x2 * (x1 * x2 * x1)^2 * x1,
        (x1, x2) -> x2 * x1^-1 * x2 * x1,
        (x1, x2) -> (x2 * x1^2)^2 * x2 * x1,
        (x1, x2) -> x2 * x1^2 * x2 * x1^-1 * x2,
        (x1, x2) -> x2 * x1 * x2,
        (x1, x2) -> x1 * x2 * x1^-2 * x2 * x1^-1,
        (x1, x2) -> (x2 * x1^-2)^2 * x2,
        (x1, x2) -> x2 * x1^-1 * x2 * x1 * x2,
        (x1, x2) -> (x2 * x1^2)^2 * x2 * x1 * x2,
        (x1, x2) -> x2 * x1^2 * x2 * x1^-1,
        (x1, x2) -> x2 * x1,
        (x1, x2) -> x2 * x1 * x2 * x1^-1 * x2 * x1^2,
        (x1, x2) -> (x2 * x1^-2)^2,
        (x1, x2) -> x1 * (x1 * x2)^2 * x1^-1,
        (x1, x2) -> x1^-2 * x2 * x1^-1,
        (x1, x2) -> x1 * (x1 * x2)^2 * x1^-1 * x2,
        (x1, x2) -> x1^-1 * (x2 * x1)^2,
        (x1, x2) -> x1^2,
        (x1, x2) -> x1^2 * x2,
        (x1, x2) -> x1 * x2 * x1^2,
        (x1, x2) -> x1^2 * x2 * x1^-1 * x2 * x1^2,
        (x1, x2) -> x1 * x2 * x1^2 * x2,
        (x1, x2) -> x2 * x1 * x2 * x1^-2 * x2 * x1,
        (x1, x2) -> x1^-1 * x2 * x1^-2,
        (x1, x2) -> x1^-1 * x2 * x1^-2 * x2,
        (x1, x2) -> x1 * x2 * x1^-2,
        (x1, x2) -> x1^-2 * x2 * x1 * x2 * x1^-1,
        (x1, x2) -> (x1 * x2 * x1)^2,
        (x1, x2) -> x2 * x1^-1 * x2 * x1^2 * x2,
        (x1, x2) -> x1^-1 * x2 * x1^-1,
        (x1, x2) -> x1^-1 * x2 * x1^-2 * x2 * x1,
        (x1, x2) -> x1 * x2 * x1^-2 * x2,
        (x1, x2) -> x1^-2 * x2 * x1 * x2 * x1^-1 * x2,
        (x1, x2) -> (x1 * x2 * x1)^2 * x2,
        (x1, x2) -> x2 * x1^-1 * x2 * x1^2,
        (x1, x2) -> (x2 * x1)^2,
        (x1, x2) -> x2 * x1^2 * x2 * x1^-1 * x2 * x1,
        (x1, x2) -> x2 * x1 * (x1 * x2)^2 * x1^-1,
        (x1, x2) -> x2 * x1^-2 * x2 * x1^-1,
        (x1, x2) -> x2 * x1 * (x1 * x2)^2 * x1^-1 * x2,
        (x1, x2) -> x2 * x1^-1 * (x2 * x1)^2,
        (x1, x2) -> x2 * x1^2,
        (x1, x2) -> x2 * x1^2 * x2,
        (x1, x2) -> x1^-2,
        (x1, x2) -> x1^2 * x2 * x1,
        (x1, x2) -> x1^-2 * x2,
        (x1, x2) -> x1 * (x1 * x2)^2,
        (x1, x2) -> x1^-1 * (x2 * x1)^2 * x1,
        (x1, x2) -> x1^-1 * (x2 * x1)^2 * x1 * x2,
        (x1, x2) -> x1^-1,
        (x1, x2) -> x1^2 * x2 * x1^2,
        (x1, x2) -> x1^-2 * x2 * x1,
        (x1, x2) -> x1 * x2 * x1^-1 * x2,
        (x1, x2) -> x1^-1 * x2 * x1 * x2 * x1^-2,
        (x1, x2) -> x1^-1 * x2 * (x1 * x2 * x1)^2,
        (x1, x2) -> x1^-1 * x2,
        (x1, x2) -> (x1^2 * x2)^2,
        (x1, x2) -> x1^-2 * x2 * x1 * x2,
        (x1, x2) -> x1 * x2 * x1^-1,
        (x1, x2) -> x2 * x1^-1 * x2 * x1^2 * x2 * x1,
        (x1, x2) -> (x1 * x2 * x1)^2 * x1,
        (x1, x2) -> (x2 * x1)^2 * x1,
        (x1, x2) -> x1 * x2 * x1^-2 * x2 * x1 * x2,
        (x1, x2) -> (x2 * x1)^2 * x1 * x2,
        (x1, x2) -> x1 * x2 * x1^-2 * x2 * x1,
        (x1, x2) -> (x1 * x2)^2 * x1^-1,
        (x1, x2) -> (x1 * x2)^2 * x1^-1 * x2,
        (x1, x2) -> x2 * x1^-2,
        (x1, x2) -> x2 * x1^2 * x2 * x1,
        (x1, x2) -> x2 * x1^-2 * x2,
        (x1, x2) -> x2 * x1 * (x1 * x2)^2,
        (x1, x2) -> x2 * x1^-1 * (x2 * x1)^2 * x1,
        (x1, x2) -> x2 * x1^-1 * (x2 * x1)^2 * x1 * x2,
    ],
)
