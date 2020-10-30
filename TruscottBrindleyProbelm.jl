using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots
using Random
using JLD2
using BSON: @save
gr()

Random.seed!(1234);

function TB(du, u, p, t)
    a, b, c, d = p
    du[1] = b*u[1].*(1-u[1]) - u[2].*((u[1].^2)./(a^2+u[1].^2)) # growth P - mortality P - grazing on P by Z
    du[2] = d*u[2].*((u[1].^2)./(a^2+u[1].^2)) - c*d*u[2] # growth Z depend on P - mortality Z
end


# Define the experimental parameter
tE=20.0f0
tspan = (0.0f0,tE)
u0 = Float32[20.0/108, 5.0/108] # devides?

p_ = Float32[0.053, 0.43, 0.024/(0.05*0.7), 0.05]
# p_ = Float32[0.053, 0.43, 0.34, 0.05] #dimless krichfall
prob = ODEProblem(TB, u0,tspan, p_)
solution = solve(prob, Rodas4(), saveat=tE/(2*tE))

plot(solution)

# Ideal data
X = Array(solution)
# Add noise to the data
println("Generate noisy data")
Xₙ = X# + Float32(1e-4)*randn(eltype(X), size(X)) # !!! signal to noise relation beachten ggf. kleiner?

L = regression_model(2, 2, 32, 2, tanh, Flux.glorot_normal)
p = initial_params(L) 

function dudt_(u, p, t)
   P,Z = u
   z = L(u,p)
   [p_[2]*P*(1-P) + z[1],#z[1]=- Z*((P^2)/(p_[1]^2+P^2)),
   p_[4]*Z*((P^2)/(p_[1]^2+P^2)) + z[2]]#z[2]=- p_[3]*p_[4]*Z]
end

prob_nn = ODEProblem(dudt_,u0, tspan, p)
sol_nn = concrete_solve(prob_nn, Rodas4(), u0, p, saveat = solution.t, maxiters=1000)

# plot true solution vs learned first guess
plot(solution)
plot!(sol_nn)


function predict(θ)
   Array(concrete_solve(prob_nn, Rodas4(), u0, θ, saveat = solution.t,
                        abstol=1e-6, reltol=1e-6,
                        sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
end

# No regularisation right now
# function loss(θ)
#    pred = predict(θ)
#    sum(abs2, Xₙ .- pred), pred
# end

function reg_loss(θ)#, λ)
   pred = predict(θ)
   sum(abs2, Xₙ .- pred) + 1e-6 * sum(θ.^ 2), pred
   # sum(abs2, Xₙ .- pred) + λ * sum(θ.^ 2), pred
end

# Test
# loss(p)
reg_loss(p)

const losses = []

# callback gets parameters + return of loss function!
callback(θ,l,pred) = begin
   push!(losses, l)
   if length(losses)%1==0
         println("Current loss after $(length(losses)) iterations: $(losses[end])")
   end
   @save "model_$(round(l, digits=3)).bson" θ
   false
end

# First train with ADAM for better convergence
res1 = DiffEqFlux.sciml_train_hack(reg_loss, p, ADAM(), cb=callback, maxiters=7)
# F_SIG => Loss(), θ, opt, cb, maxiters