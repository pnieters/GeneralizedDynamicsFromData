using GeneralizedDynamicsFromData
using OrderedCollections
using JLD2
using FileIO

experiment_name = "genetic_toggle_statistics_MSE"
repetitions = 100

weight_decay = 1e-4
η = 1e-1
η_decay_rate = 0.5
η_decay_step = 100
η_limit = 1e-4

construct_optimiser() = Flux.Optimiser(WeightDecay(weight_decay), 
                                       ExpDecay(η,η_decay_rate,η_decay_step,η_limit), 
                                       ADAM())
construct_loss(θ, y, predict) = mse_loss(θ, y, predict)

net_config = OrderedDict([
    :inputs => 2,
    :outputs => 1,
    :neurons => 16,
    :layers => 1,
    :non_lin => tanh,
    :initialization => Flux.glorot_normal
]) 

problem_one_stable = Dict([:equation => genetic_toggle_switch,
                           :parameters => Float64[1.5, 1.5, 2.0, 2.0],
                           :u0 => Float64[2.0, 4.0],
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
  filename_cb = joinpath("./data/", experiment_name*"_1s_"*string(noise)*"_all.h5")
  filename_sum = joinpath("./data/", experiment_name*"_1s_"*string(noise)*".jld2")
  callbacks_to_hdf5(callbacks, filename_cb)
  save(filename_sum, summary)
end

problem_two_stable = Dict([:equation => genetic_toggle_switch,
                           :parameters => Float64[3.5, 3.5, 2.0, 2.0],
                           :u0 => Float64[2.0, 4.0],
                           :tspan => (0.0f0, 15.0f0),
                           :ts => 0.1,
                           :solver => Tsit5,
                           :optimizer => construct_optimiser,
                           :max_iter => 1000,
                           :loss => construct_loss]
                         )

for noise in [1e-2, 5e-3, 1e-3, 1e-4]
  summary, callbacks = repeat_experiment(problem_two_stable, 
                                        net_config, 
                                        repetitions; 
                                        ε = noise, 
                                        progress=false)
  filename_cb = joinpath("./data/", experiment_name*"_1s_"*string(noise)*"_all.h5")
  filename_sum = joinpath("./data/", experiment_name*"_1s_"*string(noise)*".jld2")
  callbacks_to_hdf5(callbacks, filename_cb)
  save(filename_sum, summary)
end