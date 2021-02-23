using GeneralizedDynamicsFromData
using OrderedCollections

experiment_name = "fritzhugh_nagumo_statistics"
repetitions = 4

weight_decay = 1e-4
η = 1e-1
η_decay_rate = 0.5
η_decay_step = 100
η_limit = 1e-4

loss_weights = [1/2, 1/2]

construct_optimiser() = Flux.Optimiser(WeightDecay(weight_decay), 
                                       ExpDecay(η,η_decay_rate,η_decay_step,η_limit), 
                                       ADAM())
construct_loss(θ, y, predict) = polar_loss(θ, y, loss_weights, predict)

net_config = OrderedDict([
    :inputs => 2,
    :outputs => 1,
    :neurons => 16,
    :layers => 1,
    :non_lin => tanh,
    :initialization => Flux.glorot_normal
]) 

problem_one_stable = Dict([:equation => fritzhugh_nagumo,
                           :parameters => Float32[0.9, 0.5, 1.2, 1.25],
                           :u0 => Float32[-2.0, -0.5],
                           :tspan => (0.0f0, 5.0f0),
                           :ts => 0.1,
                           :solver => Tsit5,
                           :optimizer => construct_optimiser,
                           :max_iter => 1000,
                           :loss => construct_loss]
                      )

summary1s, callbacks1s = repeat_experiment(problem_one_stable, net_config, repetitions; ε = 1e-2)

problem_two_stable = Dict([:equation => fritzhugh_nagumo,
                           :parameters => Float32[0.9, 0.5, 1.9, 1.25],
                           :u0 => Float32[-2.0, -0.5],
                           :tspan => (0.0f0, 5.0f0),
                           :ts => 0.1,
                           :solver => Tsit5,
                           :optimizer => construct_optimiser,
                           :max_iter => 1000,
                           :loss => construct_loss]
                         )

summary2s, callbacks2s = repeat_experiment(problem_two_stable, net_config, repetitions; ε = 0.01)