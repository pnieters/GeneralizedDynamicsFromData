using GeneralizedDynamicsFromData
using OrderedCollections
using HDF5
using FileIO

experiment_name = "selkov_bifurcation_study_MSE"
repetitions = 50

weight_decay = 1e-4
η = 1e-2
η_decay_rate = 0.5
η_decay_step = 800
η_limit = 1e-4
noise = 1e-4

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

function get_parameters(k_max)

  left_branch = [0.15]
  left_stop = 0.41
  for k = 2:k_max
    push!(left_branch, left_branch[k-1]+(left_stop - left_branch[k-1])/2)
  end
  right_branch = [0.6]
  right_stop = 0.41
  for k = 2:k_max
    push!(right_branch, right_branch[k-1]-(right_branch[k-1] - right_stop)/2)
  end

  return [[0.1, p] for p in vcat(left_branch, reverse(right_branch))]

end

params = get_parameters(7)

for (id, p) in enumerate(params)

  problem = Dict([:equation => selkov,
                  :parameters => p,
                  :u0 => Float32[1.0, 1.0],
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

  fn = joinpath("./data/", experiment_name*"_$(id).h5")

  h5open(fn, "w") do file
      file["parameters"] = p
      file["losses"] = min_losses
      file["longterm_predictions"] = longterm_predictions
      file["longterm_solution"] = longterm_solution
  end

end
