using GeneralizedDynamicsFromData
using OrderedCollections
using JLD2
using FileIO

experiment_name = "genetic_toggle_statistics"
repetitions = 100

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

problem_one_stable = Dict([:equation => genetic_toggle_switch,
                           :parameters => Float32[1.5, 1.5, 2.0, 2.0],
                           :u0 => Float32[2.0, 4.0],
                           :tspan => (0.0f0, 15.0f0),
                           :ts => 0.1,
                           :solver => Tsit5,
                           :optimizer => construct_optimiser,
                           :max_iter => 1000,
                           :loss => construct_loss]
                      )

for noise in [1e-2, 5e-3, 1e-3, 1e-4]
  summary, callbacks = repeat_experiment(problem_one_stable, 
                                        net_config, 
                                        repetitions; 
                                        ε = noise, 
                                        progress=false)
  filename_cb = experiment_name*"_1s_"*string(noise)*"_all.jld2"
  filename_sum = experiment_name*"_1s_"*string(noise)*".jld2"
  save(filename_cb, Dict("callbacks" => callbacks))
  save(filename_sum, summary)
end

problem_two_stable = Dict([:equation => genetic_toggle_switch,
                           :parameters => Float32[3.5, 3.5, 2.0, 2.0],
                           :u0 => Float32[2.0, 4.0],
                           :tspan => (0.0f0, 15.0f0),
                           :ts => 0.1,
                           :solver => Tsit5,
                           :optimizer => construct_optimiser,
                           :max_iter => 1000,
                           :loss => construct_loss]
                         )

for noise in [1e-2, 5e-3, 1e-3, 1e-4]
  summary, callbacks = repeat_experiment(problem_one_stable, 
                                        net_config, 
                                        repetitions; 
                                        ε = noise, 
                                        progress=false)
  filename_cb = experiment_name*"_1s_"*string(noise)*"_all.jld2"
  filename_sum = experiment_name*"_1s_"*string(noise)*".jld2"
  save(filename_cb, Dict("callbacks" => callbacks))
  save(filename_sum, summary)
end