# SymGroupRepresentations.jl

<!-- [![Build Status](https://github.com/Yue-Zhengyuan/SymGroupRepresentations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Yue-Zhengyuan/SymGroupRepresentations.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Yue-Zhengyuan/SymGroupRepresentations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Yue-Zhengyuan/SymGroupRepresentations.jl) -->

Compute Clebsch-Gordan coefficients for small symmetric groups. Currently $S_3$, $S_4$, $S_5$ are supported. Compatibility / interoperability with [TensorKit.jl](https://github.com/Jutho/TensorKit.jl).

## Conventions

The irreps of a symmetric group $S_n$ are labelled by partitions of $n$, and arranged in "reversed dictionary order". For example, when $n = 4$, there are 5 irreps:

<center>

|   Irrep   | Partition of 4 | Dimension | Remark                  |
| :-------: | :------------: | :-------: | :---------------------- |
|   $4_1$   |      [4]       |     1     | Trivial representation  |
| $3_1 1_1$ |     [3, 1]     |     3     | Standard representation |
|   $2_2$   |     [2, 2]     |     2     |                         |
| $2_1 1_2$ |   [2, 1, 1]    |     3     |                         |
|   $1_4$   |  [1, 1, 1, 1]  |     1     | Sign representation     |

</center>

The partition of $n$ is implemented by [AbstractAlgebra.jl](https://github.com/Nemocas/AbstractAlgebra.jl). See the related docs [here](https://nemocas.github.io/AbstractAlgebra.jl/stable/ytabs/).
