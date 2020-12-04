using GeneralizedDynamicsFromData
using Statistics
using OrdinaryDiffEq
using DiffEqFlux
using Flux
using Plots
using HDF5
using OrderedCollections

experiment_name = "selkov_$(Dates.month(today()))_$(Dates.day(today()))"
repetitions = 100

net_config = OrderedDict([
    :inputs => 4,
    :outputs => 2,
    :neurons => 32,
    :layers => 2,
    :non_lin => tanh,
    :initialization => Flux.glorot_normal
]) 

problem = Dict([
    :equation => ensemble_selkov,
    :parameters => [0.1 0.6],
    :u0 => Float32[1.0, 1.0, 1.0, 1.0],
    :tE => 50.0f0,
    :tspan => (0.0f0, 50.0f0),
    :ts => 0.1,
    :solver => Vern7,
    :loss => mse_loss
])

summary = repeat_experiment(problem, net_config, repetitions)


# p_init = plot(solution, legend=false, color=:blue, vars=(1))
# plot!(p_init, solution, legend=false, color=:red, vars=(2))
# plot!(p_init, first_guess, linestyle=:dash, color=:blue, vars=(1), legend=false)
# plot!(p_init, first_guess, linestyle=:dash, color=:red, vars=(2), legend=false)

# middle_guess = concrete_solve(prob_nn, Vern7(), u0, callback.parameters[50])
# p_middle = plot(solution, legend=false, color=:blue, vars=(1))
# plot!(p_middle, solution, legend=false, color=:red, vars=(2))
# plot!(p_middle, middle_guess, linestyle=:dash, color=:blue, vars=(1), legend=false)
# plot!(p_middle, middle_guess, linestyle=:dash, color=:red, vars=(2), legend=false)

# final_guess = concrete_solve(prob_nn, Vern7(), u0, callback.parameters[end])
# p_final = plot(solution, legend=false, color=:blue, vars=(1))
# plot!(p_final, solution, legend=false, color=:red, vars=(2))
# plot!(p_final, final_guess, linestyle=:dash, color=:blue, vars=(1), legend=false)
# plot!(p_final, final_guess, linestyle=:dash, color=:red, vars=(2), legend=false)

# p_approx = plot(p_init,p_middle,p_final,layout=(3,1))
# # savefig(p_approx, "approximation_during_training_"*experiment_name*".pdf")

# σ2_ŷ = hcat([std(callback.predictions[i], dims=2).^2 for i in 1:100]...)
# p_std = plot(((σ2_ŷ .- σ2_y).^2)')
# plot!(σ2_ŷ')
# plot!(repeat([σ2_y[1]], 100))
# plot!(repeat([σ2_y[2]], 100))

# p_loss = plot(callback.losses)

# h5open("selkov_100.h5", "w") do file
#     write(file, "losses", [c.losses[end] for c in callbacks])
#     write(file, "parameters", hcat([c.parameters[end] for c in callbacks]...))
#     write(file, "predictions", hcat([c.predictions[end] for c in callbacks]...))
# end