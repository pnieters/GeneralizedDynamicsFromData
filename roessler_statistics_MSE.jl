using GeneralizedDynamicsFromData
using OrderedCollections
using JLD2
using FileIO

experiment_name = "roessler_statistics_MSE"
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
    :inputs => 3,
    :outputs => 1,
    :neurons => 16,
    :layers => 1,
    :non_lin => tanh,
    :initialization => Flux.glorot_normal
]) 

problem_period1 = Dict([:equation => roessler,
                        :parameters => Float64[0.1, 0.1, 4.0],
                        :u0 => Float64[1.0, 1.0, 1.0],
                        :tspan => (0.0f0, 10.0f0),
                        :ts => 0.1,
                        :solver => Tsit5,
                        :optimizer => construct_optimiser,
                        :max_iter => 1500,
                        :loss => construct_loss]
                      )

for noise in [5e-3, 1e-3, 1e-4, 1e-5]
  summary, callbacks = repeat_experiment(problem_period1, 
                                        net_config, 
                                        repetitions; 
                                        ε = noise, 
                                        progress=false)
  filename_cb = joinpath("./data/", experiment_name*"_p1_"*string(noise)*"_all.jld2")
  filename_sum = joinpath("./data/", experiment_name*"_p1_"*string(noise)*".jld2")
  callbacks_to_hdf5(callbacks, filename_cb)
  save(filename_sum, summary)
end

problem_period2 = Dict([:equation => roessler,
                        :parameters => Float64[0.1, 0.1, 6.0],
                        :u0 => Float64[1.0, 1.0, 1.0],
                        :tspan => (0.0f0, 10.0f0),
                        :ts => 0.1,
                        :solver => Tsit5,
                        :optimizer => construct_optimiser,
                        :max_iter => 1500,
                        :loss => construct_loss]
                      )

for noise in [5e-3, 1e-3, 1e-4, 1e-5]
  summary, callbacks = repeat_experiment(problem_period2, 
                                        net_config, 
                                        repetitions; 
                                        ε = noise, 
                                        progress=false)
  filename_cb = joinpath("./data/", experiment_name*"_p2_"*string(noise)*"_all.jld2")
  filename_sum = joinpath("./data/", experiment_name*"_p2_"*string(noise)*".jld2")
  callbacks_to_hdf5(callbacks, filename_cb)
  save(filename_sum, summary)
end