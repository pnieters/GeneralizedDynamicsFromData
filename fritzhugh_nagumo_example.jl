using GeneralizedDynamicsFromData
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra
using Plots

#### Setup parameters of the ODE, UDE system and training
parameters_one_stable = Float32[0.9, 0.5, 1.2, 1.25]
parameters_two_stable = Float32[0.9, 0.5, 1.9, 1.25]

parameters = parameters_one_stable

u₀ = Float32[-2.0, -0.5]
tspan =(0.0f0, 5.0f0)
s = 0.1

loss_weights = Float32[1/2, 1/2]
max_iters = 1000

#### Setup functions for neural network and differential equations
UA = regression_model(
    2, # input_d
    2, # output_d
    16, # hidden_d
    1, # hidden_layers
    tanh, # nonlinearity
    Flux.glorot_normal # initialization
)

real_de!, univ_de = fritzhugh_nagumo(parameters, UA)

#### solve the real ODE and create a target time series
real_prob = ODEProblem(real_de!, u₀, tspan)
real_solution = solve(real_prob, Tsit5(), saveat = s)
y = Array(real_solution)


#### set up the UDE problem and training of the neural network
θ₀ = initial_params(UA)
prob_nn = ODEProblem(univ_de, u₀, tspan, θ₀)

first_guess = concrete_solve(prob_nn, Tsit5(), u₀, θ₀, saveat=real_solution.t)
predict(θ) = Array(concrete_solve(prob_nn, Tsit5(), u₀, θ, saveat=real_solution.t))
loss(θ) = polar_loss(θ, y, loss_weights, predict)

#### train the UDE with ADAM
callback = CallbackLog(T=Float32)
opt = Flux.Optimiser(WeightDecay(1e-4), ExpDecay(1e-1,0.5,100,1e-4), ADAM())
res = DiffEqFlux.sciml_train(loss, θ₀, opt, cb=callback, maxiters=max_iters)

trained_parameters = res.minimizer
ŷ = predict(trained_parameters)
ŷ_derivs = UA(ŷ, trained_parameters)

@variables u[1:2]
b = polynomial_basis(u,5)
basis = Basis(b, u)

opt = SR3(Float32(1e-2), Float32(1e-1))
λ = exp10.(-7:0.1:5)
g(x) = x[1] > 1 ? Inf : norm(x, 2)

Ψ = SINDy(ŷ, 
          ŷ_derivs, 
          basis, 
          λ, 
          opt, 
          g = g, 
          maxiter = 50000, 
          normalize = true, 
          denoise = true, 
          convergence_error = Float32(1e-10))
println(Ψ)
print_equations(Ψ)