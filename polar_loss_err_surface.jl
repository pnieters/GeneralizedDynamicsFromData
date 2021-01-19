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

θ₀ = initial_params(UA)

prob_nn = ODEProblem(univ_de, u0, tspan, θ₀)
# first_guess = concrete_solve(prob_nn, Vern7(), u0, θ₀, saveat=solution.t)

alpha = LinRange(0, 1.0, 101)
w1_ax = LinRange(-1.0, 1.0, 1001)
w2_ax = LinRange(-1.0, 1.0, 1001)

function run()
    alpha_ax = Dict()
    @threads for (i, α) in enumerate(alpha)

        loss(θ) = polar_loss(θ, y, [1.0 - α, α], predict)
        error_surface = [loss([w1, w2, 0.0])[1] for w1 in w1_ax, w2 in w2_ax]
        alpha_ax[α] = error_surface

    end
    return alpha_ax
end

data = run()

save("ploar_loss_surface.jld2", data)