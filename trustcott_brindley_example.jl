using GeneralizedDynamicsFromData
using Optim

#### Setup parameters of the ODE, UDE system and training
# parameters_oscillation = Float32[0.053, 0.43, 0.024/(0.05*0.7), 0.05]
parameters_steadystate = Float32[0.053, 0.43, 0.34, 0.05]

parameters = parameters_steadystate

u₀ = Float32[20.0/108, 5.0/108]
tspan = (0.0f0, 30.0f0)
s = 0.1

loss_weights = Float32[1/2, 1/2]
lr = 1e-2

#### Setup functions for neural network and differential equations
UA = regression_model(
    2, # input_d
    2, # output_d
    16, # hidden_d
    1, # hidden_layers
    tanh, # nonlinearity
    Flux.glorot_normal # initialization
)

real_de!, univ_de = truscott_brindley(parameters, UA)

#### solve the real ODE and create a target time series
real_prob = ODEProblem(real_de!, u₀, tspan)
real_solution = solve(real_prob, Vern7(), saveat = s)
y = Array(real_solution)


#### set up the UDE problem and training of the neural network
θ₀ = initial_params(UA)
prob_nn = ODEProblem(univ_de, u₀, tspan, θ₀)

first_guess = concrete_solve(prob_nn, Vern7(), u₀, θ₀, saveat=real_solution.t)
predict(θ) = Array(concrete_solve(prob_nn, Vern7(), u₀, θ, saveat=real_solution.t))
loss(θ) = polar_loss(θ, y, loss_weights, predict)

#### train the UDE with ADAM
callback_adam = CallbackLog(T=Float32)
res_adam = DiffEqFlux.sciml_train(loss, θ₀, ADAM(lr), cb=callback_adam, maxiters=100)

callback_bfgs = CallbackLog(T=Float32)
res_bfgs = DiffEqFlux.sciml_train(loss, res_adam.minimizer, BFGS(initial_stepnorm=0.01), cb=callback_bfgs, maxiters = 300)

#todo: sindy optimization!