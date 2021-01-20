using GeneralizedDynamicsFromData

using OrderedCollections
using JLD2
using FileIO

using Base.Threads

net_config = OrderedDict([
    :inputs => 2,
    :outputs => 1,
    :neurons => 32,
    :layers => 0,
    :non_lin => tanh,
    :initialization => Flux.glorot_normal
]) 

equation = selkov
parameters = [0.1 0.6]
u0 = Float32[1.0, 1.0]
tE = 30.0f0
tspan = (0.0f0, 30.0f0)
ts = 0.1

UA = regression_model(values(net_config)...)
real_de, univ_de = selkov(parameters, UA)

prob = ODEProblem(real_de, u0, tspan)
solution = solve(prob, Vern7(), saveat=ts)
y = Array(solution)

θ₀ = initial_params(UA)

prob_nn = ODEProblem(univ_de, u0, tspan, θ₀)
predict(θ) = Array(concrete_solve(prob_nn, Tsit5(), u0, θ, saveat=solution.t))

alpha = LinRange(0, 1.0, 101)
w1_ax = LinRange(-1.0, 1.0, 1001)
w2_ax = LinRange(-1.0, 1.0, 1001)

cd_loss(θ) = begin
    ŷ = predict(θ)
    N = length(ŷ)
    1/N * cosine_distance(y, ŷ)
end

nld_loss(θ) = begin
    ŷ = predict(θ)
    N = length(ŷ)
    1/N * normed_ld(y, ŷ)
end

error_surface_cd = [cd_loss([w1, w2, 0.0])[1] for w1 in w1_ax, w2 in w2_ax]
error_surface_nld = [cd_loss([w1, w2, 0.0])[1] for w1 in w1_ax, w2 in w2_ax]

alpha_ax = Dict()
for α in alpha
    alpha_ax[α] = ((1-α) .* error_surface_nld) .+ (α .* error_surface_cd)
end

alpha_ax_str = Dict([(str(k), alpha_ax[k]) for k in keys(alpha_ax)])

using GLMakie
using AbstractPlotting
using AbstractPlotting.MakieLayout
using ColorSchemes

colors = ColorSchemes.berlin.colors

s = slider(LinRange(0, 1.0, 101), raw=true, camera = campixel!, start = 0.0)

data = lift(s[end][:value]) do v
    alpha_ax[v]
end

# interactions don't really work.
interactive = contour(w1_ax, w2_ax, data, linewidth = 0.2, levels=50, fillrange=true, colormap=colors)

# cm = colorlegend(scene[end], raw=true, camera=campixel!, width= (30, 540))
# scene = surface(w1_ax, w2_ax, data)
outer_padding = 30
scene, layout = layoutscene(
    outer_padding, 
    resolution = (500, 2500),
    backgroundcolor = RGBf0(0.98, 0.98, 0.98)
)
plot_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
for (plt_id, pa) in enumerate(plot_alphas)
    ax = layout[plt_id, 1] = LAxis(scene, title="$(1-pa)*NLD + $(pa)*CD")
    contour!(ax, w1_ax, w2_ax, alpha_ax[pa], linewidth=0.2, levels=50, fillrange=true, colormap=colors)
    scatter!(ax, (0.0, 0.1))
    ax.xlabel = "w_x"
    ax.ylabel = "w_y"
end