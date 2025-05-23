for I in sectorlist
    println("------------------------------------")
    println("Fusion Trees $I")
    println("------------------------------------")
    ti = time()
    N = 5
    out = ntuple(n -> randsector(I), N)
    isdual = ntuple(n -> rand(Bool), N)
    in = rand(collect(⊗(out...)))
    numtrees = count(n -> true, fusiontrees(out, in, isdual))
    while !(0 < numtrees < 30)
        out = ntuple(n -> randsector(I), N)
        in = rand(collect(⊗(out...)))
        numtrees = count(n -> true, fusiontrees(out, in, isdual))
    end
    it = @constinferred fusiontrees(out, in, isdual)
    @constinferred Nothing iterate(it)
    f = @constinferred first(it)

    # TODO: fix this but not necessary
    # @testset "Fusion tree $I: printing" begin
    #     @test eval(Meta.parse(sprint(show, f))) == f
    # end
    @timedtestset "Fusion tree $I: braiding" begin
        for in in ⊗(out...)
            for f in fusiontrees(out, in, isdual)
                for i in 1:(N - 1)
                    d1 = @constinferred TK.artin_braid(f, i)
                    @test norm(values(d1)) ≈ 1
                    d2 = empty(d1)
                    for (f1, coeff1) in d1
                        for (f2, coeff2) in TK.artin_braid(f1, i; inv=true)
                            d2[f2] = get(d2, f2, zero(coeff1)) + coeff2 * coeff1
                        end
                    end
                    @test norm(values(d2)) ≈ 1
                    for (f2, coeff2) in d2
                        if f2 == f
                            @test coeff2 ≈ 1
                        else
                            @test isapprox(coeff2, 0; atol=1000 * eps())
                        end
                    end
                end
            end
        end

        f = rand(collect(it))
        d1 = TK.artin_braid(f, 2)
        d2 = empty(d1)
        for (f1, coeff1) in d1
            for (f2, coeff2) in TK.artin_braid(f1, 3)
                d2[f2] = get(d2, f2, zero(coeff1)) + coeff2 * coeff1
            end
        end
        d1 = d2
        d2 = empty(d1)
        for (f1, coeff1) in d1
            for (f2, coeff2) in TK.artin_braid(f1, 3; inv=true)
                d2[f2] = get(d2, f2, zero(coeff1)) + coeff2 * coeff1
            end
        end
        d1 = d2
        d2 = empty(d1)
        for (f1, coeff1) in d1
            for (f2, coeff2) in TK.artin_braid(f1, 2; inv=true)
                d2[f2] = get(d2, f2, zero(coeff1)) + coeff2 * coeff1
            end
        end
        d1 = d2
        for (f1, coeff1) in d1
            if f1 == f
                @test coeff1 ≈ 1
            else
                @test isapprox(coeff1, 0; atol=1000 * eps())
            end
        end
    end
    @timedtestset "Fusion tree $I: braiding and permuting" begin
        p = tuple(randperm(N)...)
        ip = invperm(p)

        levels = ntuple(identity, N)
        d = @constinferred TK.braid(f, levels, p)
        d2 = Dict{typeof(f),valtype(d)}()
        levels2 = p
        for (f2, coeff) in d
            for (f1, coeff2) in TK.braid(f2, levels2, ip)
                d2[f1] = get(d2, f1, zero(coeff)) + coeff2 * coeff
            end
        end
        for (f1, coeff2) in d2
            if f1 == f
                @test coeff2 ≈ 1
            else
                @test isapprox(coeff2, 0; atol=1000 * eps())
            end
        end

        Af = convert(Array, f)
        Afp = permutedims(Af, (p..., N + 1))
        Afp2 = zero(Afp)
        for (f1, coeff) in d
            Afp2 .+= coeff .* convert(Array, f1)
        end
        @test Afp ≈ Afp2
    end
    @timedtestset "Fusion tree $I: insertat" begin
        N = 3
        out2 = ntuple(n -> randsector(I), N)
        in2 = rand(collect(⊗(out2...)))
        isdual2 = ntuple(n -> rand(Bool), N)
        f2 = rand(collect(fusiontrees(out2, in2, isdual2)))
        for i in 1:N
            out1 = ntuple(n -> randsector(I), N)
            out1 = Base.setindex(out1, in2, i)
            in1 = rand(collect(⊗(out1...)))
            isdual1 = ntuple(n -> rand(Bool), N)
            isdual1 = Base.setindex(isdual1, false, i)
            f1 = rand(collect(fusiontrees(out1, in1, isdual1)))

            trees = @constinferred TK.insertat(f1, i, f2)
            @test norm(values(trees)) ≈ 1

            f1a, f1b = @constinferred TK.split(f1, $i)
            @test length(TK.insertat(f1b, 1, f1a)) == 1
            @test first(TK.insertat(f1b, 1, f1a)) == (f1 => 1)

            levels = ntuple(identity, N)
            p = (i, (1:(i - 1))..., ((i + 1):N)...)
            gen = Base.Generator(TK.braid(f1, levels, p)) do (t, c)
                (t′, c′) = first(TK.insertat(t, 1, f2))
                @test c′ == one(c′)
                return t′ => c
            end
            trees2 = Dict(gen)
            trees3 = empty(trees2)
            p = (((N + 1):(N + i - 1))..., (1:N)..., ((N + i):(2N - 1))...)
            levels = ((i:(N + i - 1))..., (1:(i - 1))..., ((i + N):(2N - 1))...)
            for (t, coeff) in trees2
                for (t′, coeff′) in TK.braid(t, levels, p)
                    trees3[t′] = get(trees3, t′, zero(coeff′)) + coeff * coeff′
                end
            end
            for (t, coeff) in trees3
                @test isapprox(
                    get(trees, t, zero(coeff)), coeff; atol=1000 * eps(), rtol=1000 * eps()
                )
            end

            Af1 = convert(Array, f1)
            Af2 = convert(Array, f2)
            Af = TensorOperations.tensorcontract(
                1:(2N),
                Af1,
                [
                    1:(i - 1)
                    -1
                    N - 1 .+ ((i + 1):(N + 1))
                ],
                Af2,
                [i - 1 .+ (1:N); -1],
            )
            Af′ = zero(Af)
            for (f, coeff) in trees
                Af′ .+= coeff .* convert(Array, f)
            end
            @test Af ≈ Af′
        end
    end
    @timedtestset "Fusion tree $I: merging" begin
        N = 3
        out1 = ntuple(n -> randsector(I), N)
        in1 = rand(collect(⊗(out1...)))
        f1 = rand(collect(fusiontrees(out1, in1)))
        out2 = ntuple(n -> randsector(I), N)
        in2 = rand(collect(⊗(out2...)))
        f2 = rand(collect(fusiontrees(out2, in2)))

        @constinferred TK.merge(f1, f2, first(in1 ⊗ in2), 1)
        @test dim(in1) * dim(in2) ≈ sum(
            abs2(coeff) * dim(c) for c in in1 ⊗ in2 for μ in 1:Nsymbol(in1, in2, c) for
            (f, coeff) in TK.merge(f1, f2, c, μ)
        )

        for c in in1 ⊗ in2
            R = Rsymbol(in1, in2, c)
            for μ in 1:Nsymbol(in1, in2, c)
                trees1 = TK.merge(f1, f2, c, μ)

                # test merge and braid interplay
                trees2 = Dict{keytype(trees1),complex(valtype(trees1))}()
                trees3 = Dict{keytype(trees1),complex(valtype(trees1))}()
                for ν in 1:Nsymbol(in2, in1, c)
                    for (t, coeff) in TK.merge(f2, f1, c, ν)
                        trees2[t] = get(trees2, t, zero(valtype(trees2))) + coeff * R[μ, ν]
                    end
                end
                perm = ((N .+ (1:N))..., (1:N)...)
                levels = ntuple(identity, 2 * N)
                for (t, coeff) in trees1
                    for (t′, coeff′) in TK.braid(t, levels, perm)
                        trees3[t′] = get(trees3, t′, zero(valtype(trees3))) + coeff * coeff′
                    end
                end
                for (t, coeff) in trees3
                    @test isapprox(
                        coeff,
                        get(trees2, t, zero(coeff));
                        atol=1000 * eps(),
                        rtol=1000 * eps(),
                    )
                end

                # test via conversion
                Af1 = convert(Array, f1)
                Af2 = convert(Array, f2)
                Af0 = convert(
                    Array, FusionTree((f1.coupled, f2.coupled), c, (false, false), (), (μ,))
                )
                _Af = TensorOperations.tensorcontract(
                    1:(N + 2), Af1, [1:N; -1], Af0, [-1; N + 1; N + 2]
                )
                Af = TensorOperations.tensorcontract(
                    1:(2N + 1), Af2, [N .+ (1:N); -1], _Af, [1:N; -1; 2N + 1]
                )
                Af′ = zero(Af)
                for (f, coeff) in trees1
                    Af′ .+= coeff .* convert(Array, f)
                end
                @test Af ≈ Af′
            end
        end
    end

    N = 3
    out = ntuple(n -> randsector(I), N)
    numtrees = count(n -> true, fusiontrees((out..., map(dual, out)...)))
    while !(0 < numtrees < 100)
        out = ntuple(n -> randsector(I), N)
        numtrees = count(n -> true, fusiontrees((out..., map(dual, out)...)))
    end
    incoming = rand(collect(⊗(out...)))
    f1 = rand(collect(fusiontrees(out, incoming, ntuple(n -> rand(Bool), N))))
    f2 = rand(collect(fusiontrees(out[randperm(N)], incoming, ntuple(n -> rand(Bool), N))))

    @timedtestset "Double fusion tree $I: repartioning" begin
        for n in 0:(2 * N)
            d = @constinferred TK.repartition(f1, f2, $n)
            @test dim(incoming) ≈
                sum(abs2(coef) * dim(f1.coupled) for ((f1, f2), coef) in d)
            d2 = Dict{typeof((f1, f2)),valtype(d)}()
            for ((f1′, f2′), coeff) in d
                for ((f1′′, f2′′), coeff2) in TK.repartition(f1′, f2′, N)
                    d2[(f1′′, f2′′)] = get(d2, (f1′′, f2′′), zero(coeff)) + coeff2 * coeff
                end
            end
            for ((f1′, f2′), coeff2) in d2
                if f1 == f1′ && f2 == f2′
                    @test coeff2 ≈ 1
                    if !(coeff2 ≈ 1)
                        @show f1, f2, n
                    end
                else
                    @test isapprox(coeff2, 0; atol=1000 * eps())
                end
            end
            Af1 = convert(Array, f1)
            Af2 = permutedims(convert(Array, f2), [N:-1:1; N + 1])
            sz1 = size(Af1)
            sz2 = size(Af2)
            d1 = prod(sz1[1:(end - 1)])
            d2 = prod(sz2[1:(end - 1)])
            dc = sz1[end]
            A = reshape(
                reshape(Af1, (d1, dc)) * reshape(Af2, (d2, dc))',
                (sz1[1:(end - 1)]..., sz2[1:(end - 1)]...),
            )
            A2 = zero(A)
            for ((f1′, f2′), coeff) in d
                Af1′ = convert(Array, f1′)
                Af2′ = permutedims(convert(Array, f2′), [(2N - n):-1:1; 2N - n + 1])
                sz1′ = size(Af1′)
                sz2′ = size(Af2′)
                d1′ = prod(sz1′[1:(end - 1)])
                d2′ = prod(sz2′[1:(end - 1)])
                dc′ = sz1′[end]
                A2 +=
                    coeff * reshape(
                        reshape(Af1′, (d1′, dc′)) * reshape(Af2′, (d2′, dc′))',
                        (sz1′[1:(end - 1)]..., sz2′[1:(end - 1)]...),
                    )
            end
            @test A ≈ A2
        end
    end
    @timedtestset "Double fusion tree $I: permutation" begin
        if BraidingStyle(I) isa SymmetricBraiding
            for n in 0:(2N)
                p = (randperm(2 * N)...,)
                p1, p2 = p[1:n], p[(n + 1):(2N)]
                ip = invperm(p)
                ip1, ip2 = ip[1:N], ip[(N + 1):(2N)]

                d = @constinferred TensorKit.permute(f1, f2, p1, p2)
                @test dim(incoming) ≈
                    sum(abs2(coef) * dim(f1.coupled) for ((f1, f2), coef) in d)
                d2 = Dict{typeof((f1, f2)),valtype(d)}()
                for ((f1′, f2′), coeff) in d
                    d′ = TensorKit.permute(f1′, f2′, ip1, ip2)
                    for ((f1′′, f2′′), coeff2) in d′
                        d2[(f1′′, f2′′)] =
                            get(d2, (f1′′, f2′′), zero(coeff)) + coeff2 * coeff
                    end
                end
                for ((f1′, f2′), coeff2) in d2
                    if f1 == f1′ && f2 == f2′
                        @test coeff2 ≈ 1
                        if !(coeff2 ≈ 1)
                            @show f1, f2, p
                        end
                    else
                        @test abs(coeff2) < 1000 * eps()
                    end
                end
                Af1 = convert(Array, f1)
                Af2 = convert(Array, f2)
                sz1 = size(Af1)
                sz2 = size(Af2)
                d1 = prod(sz1[1:(end - 1)])
                d2 = prod(sz2[1:(end - 1)])
                dc = sz1[end]
                A = reshape(
                    reshape(Af1, (d1, dc)) * reshape(Af2, (d2, dc))',
                    (sz1[1:(end - 1)]..., sz2[1:(end - 1)]...),
                )
                Ap = permutedims(A, (p1..., p2...))
                A2 = zero(Ap)
                for ((f1′, f2′), coeff) in d
                    Af1′ = convert(Array, f1′)
                    Af2′ = convert(Array, f2′)
                    sz1′ = size(Af1′)
                    sz2′ = size(Af2′)
                    d1′ = prod(sz1′[1:(end - 1)])
                    d2′ = prod(sz2′[1:(end - 1)])
                    dc′ = sz1′[end]
                    A2 +=
                        coeff * reshape(
                            reshape(Af1′, (d1′, dc′)) * reshape(Af2′, (d2′, dc′))',
                            (sz1′[1:(end - 1)]..., sz2′[1:(end - 1)]...),
                        )
                end
                @test isapprox(Ap, A2; atol=1000 * eps(), rtol=1000 * eps())
            end
        end
    end
    tf = time()
    printstyled(
        "Finished fusion tree $I tests in ",
        string(round(tf - ti; sigdigits=3)),
        " seconds.";
        bold=true,
        color=Base.info_color(),
    )
    println()
end
