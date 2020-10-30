# NPZ Model

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots
using Random
# using JLD2
 gr()

 Random.seed!(1234);

 function NPZ(du, u, p, t) # unforced duffing
     a, b, g, f, kn, mn, mp, mz, Sl, N0 = p
     du[1] = Sl*(N0-u[1]) - (b*u[2])*((u[1])/(kn+u[1])) + mn*((1-g)*((a*f*u[2]^2)/(a+f*u[2]^2))*u[3] + mp*u[2] + mz*u[3]^2)
     du[2] = (b*u[2])*((u[1])/(kn+u[1])) - ((a*f*u[2]^2)/(a+f*u[2]^2))*u[3] - mp*u[2]
     du[3] = (g*u[3])*((a*f*u[2]^2)/(a+f*u[2]^2)) - mz*u[3]^2
 end

 # Define the experimental parameter
 tE=2.0f0#15.0f0# testcase #10.0f0# konvergiert nicht sauber (fittet bis kurz nach learning)
 tspan = (0.0f0,tE)
 # u0 = Float32[0.00185, 0.00355, 0.00444] #dim 0.01* ambient
 #u0 = Float32[0.185*0.3, 0.355*0.3, 0.444*0.3]
 u0 =Float32[0.01, 0.01, 0.01] # dimless 0.01*ambient

 p_ = Float32[60.0, 19.8, 0.75, 0.12288, 7.8125, 0.2, 0.9, 0.384, 0.1944, 125] # dimless
 # p_ = Float32[2.0, 0.66, 0.75, 1.0, 0.5, 0.2, 0.03, 0.2, 0.00648, 8.0] #dim
 prob = ODEProblem(NPZ, u0,tspan, p_)
 solution = solve(prob, Vern7(), saveat =tE/200)#/30)

plot(solution)

# #.... reproduziert Fig 3 a im paper für tE=30
# u1=[]
# u2=[]
# u3=[]
# for ii in 1:size(solution,2)
#     push!(u1,solution.u[ii,1][1])
#     push!(u2,solution.u[ii,1][2])
#     push!(u3,solution.u[ii,1][3])
# end

# plot(u1./0.185, label="N")
# plot!(u2./0.355, label="P")
# plot!(u3./0.444, label="Z")

#..............

 # Ideal data
 X = Array(solution)
 # Add noise to the data
 println("Generate noisy data")
 Xₙ = X# + Float32(1e-4)*randn(eltype(X), size(X)) # !!! signal to noise relation beachten ggf. kleiner?

 # Define the neueral network which learns L(x, y, y(t-τ))
 # Actually, we do not care about overfitting right now, since we want to
 # extract the derivative information without numerical differentiation.
 L = FastChain(FastDense(3, 32, tanh),FastDense(32, 32, tanh), FastDense(32, 3))
#  L = FastChain(FastDense(3, 32, tanh), FastDense(32, 3))
 p = initial_params(L)

 function dudt_(u, p, t)
    N,P,Z = u
    z = L(u,p)
    [p_[9]*(p_[10]-N) + z[1] + p_[6]*((1-p_[3])*((p_[1]*p_[4]*P^2)/(p_[1]+p_[4]*P^2))*Z + p_[7]*P + p_[8]*Z^2),#p_[1]*y,
    (p_[2]*P)*((N)/(p_[5]+N)) + z[2] - p_[7]*P,
    (p_[3]*Z)*((p_[1]*p_[4]*P^2)/(p_[1]+p_[4]*P^2)) - p_[8]*Z^2]
 end

 prob_nn = ODEProblem(dudt_,u0, tspan, p)
 sol_nn = concrete_solve(prob_nn, Rodas4(), u0, p, saveat = solution.t)#Rodas4

# plot true solution vs learned first guess
 plot(solution)
 plot!(sol_nn)

 function predict(θ)
    Array(concrete_solve(prob_nn, Rodas4(), u0, θ, saveat = solution.t,
                            abstol=1e-6, reltol=1e-6,
                         sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())))
 end

 # No regularisation right now
 function loss(θ)
    pred = predict(θ)
    sum(abs2, Xₙ .- pred), pred
 end

 # Test
 loss(p)

 const losses = []

 callback(θ,l,pred) = begin
    push!(losses, l)
    if length(losses)%50==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
 end

 # First train with ADAM for better convergence
 res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.001), cb=callback, maxiters=50)
  # res1 = DiffEqFlux.sciml_train(loss, p,ADAM(0.01), cb=callback, maxiters=300)
 # Plot the losses
 plot(losses, yaxis = :log, xaxis = :log, xlabel = "Iterations", ylabel = "Loss")
