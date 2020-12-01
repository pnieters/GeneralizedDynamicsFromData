module GeneralizedDynamicsFromData

using OrdinaryDiffEq
using LinearAlgebra
using Optim
using DiffEqFlux
using Flux
using Plots
using StatsPlots
using Statistics

export regression_model, reshape_all_parameters, reshape_parameters, restructure_parameters 
export CallbackLog
include("utils.jl")

export truscott_brindley, selkov, selkov_sigma2
include("eqlib.jl")

export plot_weights
include("plotlib.jl")

end # module
