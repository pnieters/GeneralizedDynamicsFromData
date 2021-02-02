using GeneralizedDynamicsFromData

using GLMakie
# using CairoMakie
using AbstractPlotting
# using AbstractPlotting.MakieLayout
using ColorSchemes

# Bifurcation Parameter α
α = LinRange(-0.5, 0.5, 1001)
u₀ = Float32[1.0, 1.0]
tmax = 30.0f0
t₀ = 0.0f0
t_step = 0.1

function fake_selkov(a1, a2, b)
    # a1 a2 -> unterschiedlichen vorzeichen machen probleme

    real_de = (du, u, p, t) -> begin
        # dx/dt = -x + a1*y + x^2 y 
        du[1] = -u[1] + a1*u[2] + (u[1].^2).*u[2]
        # dx/dt = b - a2*y - x^2 y
        du[2] = b - a2*u[2] - (u[1].^2)*u[2]
    end

    return real_de
end

ode_collection = [fake_selkov(a, 0.1, 0.6) for a in α]
ode_problems = [ODEProblem(ode, u₀, (t₀, tmax)) for ode in ode_collection]
ode_solutions = [Array(solve(ode, Vern7(), saveat=t_step)) for ode in ode_problems]

N = length(α)
M = size(ode_solutions[1],2)

loss_matrix_mse = Array{Float32, 2}(undef, N, N)
loss_matrix_cd = Array{Float32, 2}(undef, N, N)
loss_matrix_nld = Array{Float32, 2}(undef, N, N)

for (i, j) in Iterators.product(1:N, 1:N)
    if length(ode_solutions[i]) != length(ode_solutions[j])
        loss_matrix_mse[i,j] = NaN
        loss_matrix_cd[i,j] = NaN
        loss_matrix_nld[i,j] = NaN
    else
        loss_matrix_mse[i,j] = sum(abs2, ode_solutions[i] .- ode_solutions[j])
        loss_matrix_cd[i,j] = 1/N * sum([cosine_distance(ode_solutions[i][:,k], ode_solutions[j][:,k]) for k in 1:M])
        loss_matrix_nld[i,j] = 1/N * sum([normed_ld(ode_solutions[i][:,k], ode_solutions[j][:,k]) for k in 1:M])
    end
end

# Here comes a lot of plotting!
# |-->
# |--> Slider for α_ref and α_compare by idx?

# cmap = ColorSchemes.berlin.colors
cmap = ColorSchemes.oslo.colors

fig = Figure(
    outer_padding = 30,
    resolution = (1500, 1000),
    backgroundcolor = RGBf0(0.9, 0.9, 0.9)
)

# ref_α = 0.1
ref_α_id = 610
ref_α = α[ref_α_id]
half_window = 10

# for some reason sliders seem mighty buggy
# reference_slider = Slider(fig[1,1], range=1:N, startvalue=1)
# ref_α = lift(reference_slider.value) do v
#     α[v]
# end

# comp_slider = Slider(fig[1,2], range=1:N, startvalue=1)
# comp_α = lift(comp_slider.value) do v
#     α[v]
# end



ax_mse_surface = fig[1,1] = Axis(fig, title="MSE")
heatmap!(ax_mse_surface, α, α, log.(loss_matrix_mse), colormap=cmap)
vlines!(ax_mse_surface, [ref_α], color=:red)
hlines!(ax_mse_surface, [ref_α], color=:red)
# scatter!(ax_mse_surface, ([ref_α], [comp_α]), markersize=10)
# limits!(ax_mse_surface, α[1], α[end], α[1], α[end])

ax_mse_cut = fig[2,1] = Axis(fig)
lines!(ax_mse_cut, α, loss_matrix_mse[:,ref_α_id])

ax_cd_surface = fig[1,2] = Axis(fig, title="Cosine Distance")
heatmap!(ax_cd_surface, α, α, loss_matrix_cd, colormap=cmap)
vlines!(ax_cd_surface, [ref_α], color=:red)
hlines!(ax_cd_surface, [ref_α], color=:red)
# scatter!(ax_cd_surface, ([ref_α], [comp_α]), markersize=10)
# limits!(ax_cd_surface, α[1], α[end], α[1], α[end])

ax_cd_cut = fig[2,2] = Axis(fig)
lines!(ax_cd_cut, α, loss_matrix_cd[:,ref_α_id])

ax_nld_surface = fig[1,3] = Axis(fig, title="Normalized Length Distance")
heatmap!(ax_nld_surface, α, α, loss_matrix_nld, colormap=cmap)
vlines!(ax_nld_surface, [ref_α], color=:red)
hlines!(ax_nld_surface, [ref_α], color=:red)
# scatter!(ax_nld_surface, ([ref_α], [comp_α]), markersize=10)
# limits!(ax_nld_surface, α[1], α[end], α[1], α[end])

ax_nld_cut = fig[2,3] = Axis(fig)
lines!(ax_nld_cut, α, loss_matrix_nld[:,ref_α_id])

# fig2 = Figure(
#     outer_padding = 30,
#     resolution = (1500, 1000),
#     backgroundcolor = RGBf0(0.9, 0.9, 0.9)
# )

# ax = fig[1,1] = Axis(fig2)
# lines!(ax, α[ref_α_id-120:ref_α_id+100], loss_matrix_mse[ref_α_id-120:ref_α_id+100, ref_α_id])