using GeneralizedDynamicsFromData
using Plots
using HDF5
using OrderedCollections
using Dates


experiment_name = "selkov_grid_$(Dates.month(today()))_$(Dates.day(today()))"

reps = 2 #100
neurons = [8] #[2^i for i in 1:10]
layers = [1]#[i for i in 1:10]

grid_config = [OrderedDict([
    :inputs => 2,
    :outputs => 1,
    :neurons => N,
    :layers => L,
    :non_lin => tanh,
    :initialization => Flux.glorot_normal
]) for (N, L) in Iterators.product(neurons, layers)]

problem = Dict([
    :equation => selkov,
    :parameters => [0.1 0.6],
    :u0 => Float32[1.0, 1.0],
    :tE => 50.0f0,
    :tspan => (0.0f0, 50.0f0),
    :ts => 0.1,
    :solver => Vern7,
    :loss => mse_loss
])

result = grid_experiment(problem, grid_config, reps)
