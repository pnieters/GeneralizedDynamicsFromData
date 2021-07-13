using GeneralizedDynamicsFromData
using OrderedCollections
using HDF5
using FileIO

experiment_name = "selkov_iv_study"
repetitions = 50

weight_decay = 1e-4
η = 1e-2
η_decay_rate = 0.5
η_decay_step = 800
η_limit = 1e-4
noise = 1e-4

loss_weights = [0.2, 0.8]

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


initial_values = [ Float64[1.0 ,1.0],
                   Float64[0.5 ,1.0],
                   Float64[0.0 ,1.0],
                   Float64[0.25,0.5],
                   Float64[0.25,3.0],
                   Float64[0.5 ,6.0] ]

params = [ [0.1, 0.6],    # oscillation
           [0.1, 0.15] ]   # steadystate

for (id, u0) in enumerate(initial_values)

  problem = Dict([:equation => selkov,
                  :parameters => params[1],
                  :u0 => u0,
                  :tspan => (0.0f0, 15.0f0),
                  :ts => 0.1,
                  :solver => Tsit5,
                  :optimizer => construct_optimiser,
                  :max_iter => 3000,
                  :loss => construct_loss
                 ])

  summary, _ = repeat_experiment(problem, 
                                 net_config, 
                                 repetitions; 
                                 longterm = 60.0,
                                 ε = noise, 
                                 progress=false)

  min_losses = convert(Array{Float64,1}, summary["losses"])
  longterm_predictions = cat([sol for sol in summary["longterm_predictions"]]...; dims=3)
  longterm_solution = cat([sol for sol in summary["longterm_solution"]]...; dims=3)

  fn = joinpath("./data/", experiment_name*"_oscillation_"*"$(id).h5")

  h5open(fn, "w") do file
      file["parameters"] = params[1]
      file["u0"] = u0
      file["losses"] = min_losses
      file["longterm_predictions"] = longterm_predictions
      file["longterm_solution"] = longterm_solution
  end

end

for (id, u0) in enumerate(initial_values)

  problem = Dict([:equation => selkov,
                  :parameters => params[2],
                  :u0 => u0,
                  :tspan => (0.0f0, 15.0f0),
                  :ts => 0.1,
                  :solver => Tsit5,
                  :optimizer => construct_optimiser,
                  :max_iter => 3000,
                  :loss => construct_loss
                 ])

  summary, _ = repeat_experiment(problem, 
                                 net_config, 
                                 repetitions; 
                                 longterm = 60.0,
                                 ε = noise, 
                                 progress=false)

  min_losses = convert(Array{Float64,1}, summary["losses"])
  longterm_predictions = cat([sol for sol in summary["longterm_predictions"]]...; dims=3)
  longterm_solution = cat([sol for sol in summary["longterm_solution"]]...; dims=3)

  fn = joinpath("./data/", experiment_name*"_steadystate_"*"$(id).h5")

  h5open(fn, "w") do file
      file["parameters"] = params[2]
      file["u0"] = u0
      file["losses"] = min_losses
      file["longterm_predictions"] = longterm_predictions
      file["longterm_solution"] = longterm_solution
  end

end
