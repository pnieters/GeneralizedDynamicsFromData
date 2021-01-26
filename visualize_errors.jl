using GeneralizedDynamicsFromData

# using GLMakie
# using AbstractPlotting
# using AbstractPlotting.MakieLayout
# using ColorSchemes

# Bifurcation Parameter α
α = LinRange(-0.5, 0.5, 101)
u₀ = Float32[1.0, 1.0]
tmax = 30.0f0
t₀ = 0.0f0
t_step = 0.1

ode_collection = [selkov([a, 0.6], (u,p)->0)[1] for a in α]
ode_problems = [ODEProblem(ode, u₀, (t₀, tmax)) for ode in ode_collection]
ode_solutions = [Array(solve(ode, Vern7(), saveat=t_step)) for ode in ode_problems]

N = length(α)
M = size(ode_solutions[1],2)

loss_matrix_mse = Array{Float32, 2}(undef, N, N)
loss_matrix_cd = Array{Float32, 2}(undef, N, N)
loss_matrix_nld = Array{Float32, 2}(undef, N, N)

for (i, j) in Iterators.product(1:N, 1:N)
    loss_matrix_mse[i,j] = sum(abs2, ode_solutions[i] .- ode_solutions[j])
    loss_matrix_cd[i,j] = 1/N * sum([cosine_distance(ode_solutions[i][:,k], ode_solutions[j][:,k]) for k in 1:M])
    loss_matrix_nld[i,j] = 1/N * sum([normed_ld(ode_solutions[i][:,k], ode_solutions[j][:,k]) for k in 1:M])
end

# Here comes a lot of plotting!
# |-->
# |--> Slider for α_ref and α_compare