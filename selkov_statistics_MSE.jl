using GeneralizedDynamicsFromData
using OrderedCollections
using JLD2
using FileIO

experiment_name = "selkov_statistics_MSE"
repetitions = 100

weight_decay = 1e-4
η = 1e-2
η_decay_rate = 0.5
η_decay_step = 800
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

problem_oscillation = Dict([:equation => selkov,
                           :parameters => Float32[0.1, 0.6],
                           :u0 => Float32[1.0, 1.0],
                           :tspan => (0.0f0, 15.0f0),
                           :ts => 0.1,
                           :solver => Tsit5,
                           :optimizer => construct_optimiser,
                           :max_iter => 3000,
                           :loss => construct_loss]
                      )

for noise in [1e-2, 5e-3, 1e-3, 1e-4]
  summary, callbacks = repeat_experiment(problem_oscillation, 
                                        net_config, 
                                        repetitions; 
                                        ε = noise, 
                                        progress=false)
  filename_cb = joinpath("./data", experiment_name*"_oscillation_"*string(noise)*"_all.h5")
  filename_sum = joinpath("./data", experiment_name*"_oscillation_"*string(noise)*".jld2")
  callbacks_to_hdf5(callbacks, filename_cb)
  save(filename_sum, summary)
end

problem_steadystate = Dict([:equation => selkov,
                           :parameters => Float32[0.1, 0.15],
                           :u0 => Float32[1.0, 1.0],
                           :tspan => (0.0f0, 15.0f0),
                           :ts => 0.1,
                           :solver => Tsit5,
                           :optimizer => construct_optimiser,
                           :max_iter => 1000,
                           :loss => construct_loss]
                         )

for noise in [1e-2, 5e-3, 1e-3, 1e-4]
  summary, callbacks = repeat_experiment(problem_steadystate, 
                                        net_config, 
                                        repetitions; 
                                        ε = noise, 
                                        progress=false)
  filename_cb = joinpath("./data", experiment_name*"_steadystate_"*string(noise)*"_all.h5")
  filename_sum = joinpath("./data", experiment_name*"_steadystate_"*string(noise)*".jld2")
  callbacks_to_hdf5(callbacks, filename_cb)
  save(filename_sum, summary)
end