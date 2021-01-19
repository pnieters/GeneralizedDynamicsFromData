using GeneralizedDynamicsFromData

using OrderedCollections
using JLD2
using FileIO

using Base.Threads

net_config = OrderedDict([
    :inputs => 2,
    :outputs => 1,
    :neurons => 32,
    :layers => 0,
    :non_lin => tanh,
    :initialization => Flux.glorot_normal
]) 

equation = selkov
parameters = [0.1 0.6]
u0 = Float32[1.0, 1.0]
tE = 30.0f0
tspan = (0.0f0, 30.0f0)
ts = 0.1

UA = regression_model(values(net_config)...)
real_de, univ_de = selkov(parameters, UA)

prob = ODEProblem(real_de, u0, tspan)
solution = solve(prob, Vern7(), saveat=ts)
y = Array(solution)

θ₀ = initial_params(UA)

prob_nn = ODEProblem(univ_de, u0, tspan, θ₀)
predict(θ) = Array(concrete_solve(prob_nn, Tsit5(), u0, θ, saveat=solution.t))

alpha = LinRange(0, 1.0, 101)
w1_ax = LinRange(-1.0, 1.0, 1001)
w2_ax = LinRange(-1.0, 1.0, 1001)

cd_loss(θ) = begin
    ŷ = predict(θ)
    N = length(ŷ)
    1/N * cosine_distance(y, ŷ)
end

nld_loss(θ) = begin
    ŷ = predict(θ)
    N = length(ŷ)
    1/N * normed_ld(y, ŷ)
end

error_surface_cd = [cd_loss([w1, w2, 0.0])[1] for w1 in w1_ax, w2 in w2_ax]
error_surface_nld = [cd_loss([w1, w2, 0.0])[1] for w1 in w1_ax, w2 in w2_ax]

alpha_ax = Dict()
for α in alpha
    alpha_ax[α] = ((1-α) .* error_surface_nld) .+ (α .* error_surface_cd)
end