using GeneralizedDynamicsFromData

#### Setup parameters of the ODE, UDE system and training
parameters_one_stable = Float32[0.9, 0.5, 1.2, 1.25]
parameters_two_stable = Float32[0.9, 0.5, 1.9, 1.25]

parameters = parameters_one_stable

u₀ = Float32[-2.0, -0.5]
tspan =(0.0f0, 5.0f0)
s = 0.1

loss_weights = Float32[1/2, 1/2]
λ = 1e-7
lr = 1e-2
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
loss(θ) = polar_loss(θ, y, loss_weights, λ, predict)

#### train the UDE with ADAM
callback = CallbackLog(T=Float32)
res_adam = DiffEqFlux.sciml_train(loss, θ₀, ADAM(lr), cb=callback, maxiters=max_iters)

#todo: sindy optimization!