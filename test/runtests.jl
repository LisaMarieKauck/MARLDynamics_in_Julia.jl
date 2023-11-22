using MARLDynamics_in_Julia
using Test

using UtilitiesHelpers

@testset "MARLDynamics_in_Julia.jl" begin
    vec_singular = make_variable_vector(0.9, 5)
    vec_consistent_length = make_variable_vector([0.1, 0.1, 0.1, 0.1, 0.1], 5)

    @test length(vec_singular) == 5
    @test vec_singular == [0.9, 0.9, 0.9, 0.9, 0.9]
    @test length(vec_consistent_length) == 5
    @test vec_consistent_length == [0.1, 0.1, 0.1, 0.1, 0.1]
    @test_throws AssertionError make_variable_vector([0.1, 0.1, 0.1, 0.1], 5)
end

@testset "MARLDynamics_in_Julia.jl" begin
    Tkk = rand(4,4)
    Tkk = Tkk/sum(Tkk)

    @test sum(Tkk) == 1

    SD_Tkk = compute_stationarydistribution(Tkk)
    @test sum(SD_Tkk[:, 2:4] .== -10.) == 12
end
