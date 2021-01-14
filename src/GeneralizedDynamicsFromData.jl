module GeneralizedDynamicsFromData

using Reexport

using LinearAlgebra
using Optim
using Plots
using StatsPlots
using Statistics
using OrderedCollections
using FileIO

using JLD2

@reexport using OrdinaryDiffEq
@reexport using DiffEqFlux
@reexport using Flux

export regression_model, reshape_all_parameters, reshape_parameters, restructure_parameters 
export CallbackLog
export repeat_experiment, grid_experiment
export InitializationLoader
include("utils.jl")

export truscott_brindley, truscott_brindley1, selkov, selkov_sigma2, ensemble_selkov
include("eqlib.jl")

export mse_loss, normed_ld_loss, cosine_distance_loss, combined_loss
include("losslib.jl")

export plot_weights
include("plotlib.jl")

end # module
