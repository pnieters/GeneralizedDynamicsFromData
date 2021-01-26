using GeneralizedDynamicsFromData

using OrderedCollections

using GLMakie # CairoMakie does VectorGraphics; but doesn't do 3D?
using AbstractPlotting
using AbstractPlotting.MakieLayout

net_config = OrderedDict([
    :inputs => 2,
    :outputs => 1,
    :neurons => 32,
    :layers => 2,
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

y = Array(solution)
predict(θ) = Array(concrete_solve(prob_nn, Vern7(), u0, θ, saveat=solution.t))


loss_phase1(θ) = polar_loss(θ, y, [1/11, 10/11], predict)
callback_phase1 = CallbackLog(T=Float32)
res_phase1 = DiffEqFlux.sciml_train(loss_phase1, θ₀, ADAM(1e-2), cb=callback_phase1, maxiters=1_000; progress=true)

θ₁ = callback_phase1.parameters[end]
loss_phase2(θ) = polar_loss(θ, y, [1/2, 1/2],predict)
callback_phase2 = CallbackLog(T=Float32)
res_phase2 = DiffEqFlux.sciml_train(loss_phase2, θ₁, ADAM(1e-3), cb=callback_phase2, maxiters=500; progress=true)