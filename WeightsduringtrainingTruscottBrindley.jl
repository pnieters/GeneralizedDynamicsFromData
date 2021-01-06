# HOPFBIFURKATION Truscott Brindley
# weight eintwicklung first guess zu final


cd("d:\\ScientificData\\ApproximateDynamics\\FFTstudy\\GeneralizedDynamicsFromData-master")
# using GeneralizedDynamicsFromData
using OrdinaryDiffEq
using LinearAlgebra
using Optim
using DiffEqFlux
using Flux
using StatsPlots
using Statistics

export regression_model, reshape_all_parameters, reshape_parameters, restructure_parameters
export CallbackLog
include("src\\utils.jl")

export truscott_brindley
include("src\\_eqlib.jl")

export plot_weights
include("src\\plotlib.jl")
#-----------------------------------

parm = [0.053, 0.43, 0.024/(0.05*0.7), 0.05] # oszillation

UA = regression_model(2, 1, 0, 0, tanh, Flux.glorot_normal)

# don't think this makes sense
real_de, univ_de = truscott_brindley1(parm, UA)
tE=300.0f0#
tspan = (0.0f0,tE)
u0 = Float32[20.0/108, 5.0/108]
ts = 0.1

prob = ODEProblem(real_de, u0, tspan)
solution = solve(prob, Rodas4(), saveat=ts)# Rodas4 for truscott_brindley

plot(solution)

X  = Array(solution)
XT = solution.t
#
# pp_loss = []
# pp_preds = []
# pp_weights =[]
#
# for kk=1:200
#     println("run $(kk)")
#     θ₀=[]
#     UA=[]
#     res2=[]
#
#     UA = regression_model(2, 1, 0, 0, tanh, Flux.glorot_normal)
#     real_de, univ_de = selkov(parameters, UA)
#     # # first guess parameter
    θ₀ = initial_params(UA)
    prob_nn = ODEProblem(univ_de, u0, tspan, θ₀)
    first_guess = concrete_solve(prob_nn, Rodas4(), u0, θ₀, saveat=solution.t) # Rodas4
    plot(first_guess)
    plot!(solution)
#
    predict(θ) = Array(concrete_solve(prob_nn, Rodas4(), u0, θ, saveat=solution.t)) # Rodas4
    loss(θ) = begin
        pred = predict(θ)
        sum(abs2, X .- pred), pred
    end
#
    callback = CallbackLog(T=Float32)
    res2 = DiffEqFlux.sciml_train(loss, θ₀, ADAM(1e-4), cb=callback, maxiters=300)
p_loss = plot(callback.losses)
final_guess = concrete_solve(prob_nn, Rodas4(), u0, callback.parameters[end])
p_final = plot(solution, legend=false, color=:blue, vars=(1))
plot!(p_final, solution, legend=false, color=:red, vars=(2))
plot!(p_final, final_guess, linestyle=:dash, color=:blue, vars=(1), legend=false)
plot!(p_final, final_guess, linestyle=:dash, color=:red, vars=(2), legend=false)

    #
#     push!(pp_loss, callback.losses)
#     push!(pp_preds,callback.predictions)
#     push!(pp_weights, callback.parameters)
# end
#