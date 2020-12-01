# How could an expeirment using more shared functions look like?
# Is seperating out the equations into a library a good idea??

# no package, just a bunch of 'outsource' code is fine!
using GeneralizedDynamicsFromData
using OrdinaryDiffEq
using DiffEqFlux
using Flux
using Plots

experiment_name = "lin_and_log_slow"

parameters = [0.053, 0.43, 0.024/(0.05*0.7), 0.05]
# parameters = Float32[0.053, 0.43, 0.34, 0.05] #dimless krichfall
UA = regression_model(2, 2, 32, 2, tanh, Flux.glorot_normal)

# don't think this makes sense
real_de, univ_de = truscott_brindley(parameters, UA)
u0 = Float32[20.0/108, 5.0/108]
tE = 40.0f0
tspan = (0.0f0, tE)

prob = ODEProblem(real_de, u0, tspan)
solution = solve(prob, Rodas4(), saveat=tE/(2*tE))

plot(solution)

X  = Array(solution)
θ₀ = initial_params(UA)

prob_nn = ODEProblem(univ_de, u0, tspan, θ₀)

first_guess = concrete_solve(prob_nn, Rodas4(), u0, θ₀, saveat=solution.t)
plot(first_guess)

predict(θ) = Array(concrete_solve(prob_nn, Rodas4(), u0, θ, saveat=solution.t))
loss(θ) = begin
    pred = predict(θ)
    sum(abs2, X .- pred), pred
end

callback = CallbackLog(T=Float32)

res1 = DiffEqFlux.sciml_train(loss, θ₀, ADAM(1e-6), cb=callback, maxiters=300)

pw = plot_weights(callback.parameters, 2, 2, 32, 2)
savefig(pw, "figures/weights_during_training_"*experiment_name*".pdf")

p_loss = plot(callback.losses)
savefig(p_loss, "figures/loss_curve_"*experiment_name*".pdf")

# Plot approximations during the training time!
p_init = plot(solution, legend=false, color=:blue, vars=(1))
plot!(p_init, solution, legend=false, color=:red, vars=(2))
plot!(p_init, first_guess, linestyle=:dash, color=:blue, vars=(1), legend=false)
plot!(p_init, first_guess, linestyle=:dash, color=:red, vars=(2), legend=false)

middle_guess = concrete_solve(prob_nn, Rodas4(), u0, callback.parameters[50])
p_middle = plot(solution, legend=false, color=:blue, vars=(1))
plot!(p_middle, solution, legend=false, color=:red, vars=(2))
plot!(p_middle, middle_guess, linestyle=:dash, color=:blue, vars=(1), legend=false)
plot!(p_middle, middle_guess, linestyle=:dash, color=:red, vars=(2), legend=false)

final_guess = concrete_solve(prob_nn, Rodas4(), u0, callback.parameters[end])
p_final = plot(solution, legend=false, color=:blue, vars=(1))
plot!(p_final, solution, legend=false, color=:red, vars=(2))
plot!(p_final, final_guess, linestyle=:dash, color=:blue, vars=(1), legend=false)
plot!(p_final, final_guess, linestyle=:dash, color=:red, vars=(2), legend=false)

p_approx = plot(p_init,p_middle,p_final,layout=(3,1))
savefig(p_approx, "figures/approximation_during_training_"*experiment_name*".pdf")

# simulate trained result for longer! Does it generalize?
tspan = (0.0f0, 200.0f0)
prob = ODEProblem(real_de, u0, tspan)
solution = solve(prob, Rodas4())

prob_nn = ODEProblem(univ_de, u0, tspan, θ₀)
first_guess = concrete_solve(prob_nn, Rodas4(), u0, θ₀)

p_init = plot(solution, legend=false, color=:blue, vars=(1))
plot!(p_init, solution, legend=false, color=:red, vars=(2))
plot!(p_init, first_guess, linestyle=:dash, color=:blue, vars=(1), legend=false)
plot!(p_init, first_guess, linestyle=:dash, color=:red, vars=(2), legend=false)
plot!(p_init, [20.0], seriestype=:vline, color=:black, linestyle=:dot)

middle_guess = concrete_solve(prob_nn, Rodas4(), u0, callback.parameters[50])
p_middle = plot(solution, legend=false, color=:blue, vars=(1))
plot!(p_middle, solution, legend=false, color=:red, vars=(2))
plot!(p_middle, middle_guess, linestyle=:dash, color=:blue, vars=(1), legend=false)
plot!(p_middle, middle_guess, linestyle=:dash, color=:red, vars=(2), legend=false)
plot!(p_middle, [20.0], seriestype=:vline, color=:black, linestyle=:dot)

final_guess = concrete_solve(prob_nn, Rodas4(), u0, callback.parameters[end])
p_final = plot(solution, legend=false, color=:blue, vars=(1))
plot!(p_final, solution, legend=false, color=:red, vars=(2))
plot!(p_final, final_guess, linestyle=:dash, color=:blue, vars=(1), legend=false)
plot!(p_final, final_guess, linestyle=:dash, color=:red, vars=(2), legend=false)
plot!(p_final, [20.0], seriestype=:vline, color=:black, linestyle=:dot)

p_approx = plot(p_init,p_middle,p_final,layout=(3,1))
savefig(p_approx, "figures/approximation_longer_"*experiment_name*".pdf")