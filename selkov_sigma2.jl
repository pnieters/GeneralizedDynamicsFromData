using GeneralizedDynamicsFromData
using Statistics
using OrdinaryDiffEq
using DiffEqFlux
using Flux
using Plots

# This may work!, TODO:
# 1. MSE on the "real" variables + λ * (MSE on inst. variance)
# 2. the neural network should only see x, y -- not the derived variables!

experiment_name = "selkov_sigma2_3"

parameters = [0.1, 0.6]
UA = regression_model(2,1,32,2,tanh,Flux.glorot_normal)

real_de, univ_de = selkov_sigma2(parameters, UA, 0.1)
u0 = Float32[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
tE = 50.0f0
tspan = (0.0f0, tE)
ts = 0.1

prob = ODEProblem(real_de, u0, tspan)
solution = solve(prob, Tsit5(), saveat=ts)

px = plot()
plot!(px, solution, color=:red, vars=(1))
plot!(px, solution, linestyle=:dash, color=:red, vars=(3))
plot!(px, solution, linestyle=:dash, color=:orange, vars=(5))

py = plot()
plot!(py, solution, color=:blue, vars=(2))
plot!(py, solution, linestyle=:dash, color=:blue, vars=(4))
plot!(py, solution, linestyle=:dash, color=:purple, vars=(6))

p_vars = plot(px, py, layout=(2,1))
savefig(p_vars, "dynamics_"*experiment_name*".pdf")


y = Array(solution)[[1,2,5,6],:]
θ₀ = initial_params(UA)
λ = 1.0f0

prob_nn = ODEProblem(univ_de, u0, tspan, θ₀)
first_guess = concrete_solve(prob_nn, Tsit5(), u0, θ₀, saveat=solution.t)

predict(θ) = Array(concrete_solve(prob_nn, Tsit5(), u0, θ, saveat=solution.t))
loss(θ) = begin
    ŷ = predict(θ)[[1,2,5,6],:] # 3 & 4 are running averages: not interesting!
    loss = sum(abs2, [0.1f0;0.1f0;λ^2;λ^2] .* (y .- ŷ)), ŷ
end

callback = CallbackLog(T=Float32)
res = DiffEqFlux.sciml_train(loss, θ₀, ADAM(1e-3), cb=callback, maxiters=300)

p_init = plot(solution, legend=false, color=:blue, vars=(1))
plot!(p_init, solution, legend=false, color=:red, vars=(2))
plot!(p_init, first_guess, linestyle=:dash, color=:blue, vars=(1), legend=false)
plot!(p_init, first_guess, linestyle=:dash, color=:red, vars=(2), legend=false)

middle_guess = concrete_solve(prob_nn, Tsit5(), u0, callback.parameters[150])
p_middle = plot(solution, legend=false, color=:blue, vars=(1))
plot!(p_middle, solution, legend=false, color=:red, vars=(2))
plot!(p_middle, middle_guess, linestyle=:dash, color=:blue, vars=(1), legend=false)
plot!(p_middle, middle_guess, linestyle=:dash, color=:red, vars=(2), legend=false)

final_guess = concrete_solve(prob_nn, Tsit5(), u0, callback.parameters[end])
p_final = plot(solution, legend=false, color=:blue, vars=(1))
plot!(p_final, solution, legend=false, color=:red, vars=(2))
plot!(p_final, final_guess, linestyle=:dash, color=:blue, vars=(1), legend=false)
plot!(p_final, final_guess, linestyle=:dash, color=:red, vars=(2), legend=false)

p_approx = plot(p_init,p_middle,p_final,layout=(3,1))
savefig(p_approx, "approximation_during_training_"*experiment_name*".pdf")

# σ2_ŷ = hcat([std(callback.predictions[i], dims=2).^2 for i in 1:100]...)
# p_std = plot(((σ2_ŷ .- σ2_y).^2)')
# plot!(σ2_ŷ')
# plot!(repeat([σ2_y[1]], 100))
# plot!(repeat([σ2_y[2]], 100))

p_loss = plot(callback.losses)
savefig(p_loss, "loss_"*experiment_name*".pdf")
