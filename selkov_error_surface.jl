using GeneralizedDynamicsFromData
using OrderedCollections
using Statistics

using GLMakie # CairoMakie does VectorGraphics; but doesn't do 3D?
using AbstractPlotting
using AbstractPlotting.MakieLayout

experiment_name = "selkov_error_surface"
repetitions = 100

net_config = OrderedDict([
    :inputs => 2,
    :outputs => 1,
    :neurons => 32,
    :layers => 0,
    :non_lin => tanh,
    :initialization => Flux.glorot_normal
]) 

UA = regression_model(values(net_config)...)
parameters = [0.1, 0.6]
u0 = Float32[1.0, 1.0]
tspan = (0.0f0, 50.0f0)
ts = 0.1

real_de, univ_de = selkov(parameters, UA)

prob = ODEProblem(real_de, u0, tspan)
solution = solve(prob, Tsit5(), saveat=ts)
y = Array(solution)

prob_nn = ODEProblem(univ_de, u0, tspan, [0.0, 0.0, 0.0])
predict(θ) = Array(concrete_solve(prob_nn, Tsit5(), u0, θ, saveat=solution.t))
loss(θ) = combined_loss(θ, y, predict)

w1_ax = LinRange(-0.5, 0.5, 1001)
w2_ax = LinRange(-0.5, 0.5, 1001)

error_surface = [loss([w1, w2, 0.0])[1] for w1 in w1_ax, w2 in w2_ax]

# make this somewhat digistable for the plot
# transformed_surface = 1/std(error_surface) .* (error_surface .- mean(error_surface))

s = surface(w1_ax, w2_ax, error_surface)
scatter!(s, (0.0, 0.1, 0.0))

# s is a 3-D scene, only saveable as image with camera.


# # cut slices through the minimum

# x_range = LinRange(-0.05, 0.05, 1001)
# error_x = [loss([x, 0.1, 0.0])[1] for x in x_range]

# y_range = LinRange(0.05, 0.15, 1001)
# error_y = [loss([0.0, y, 0.0])[1] for y in y_range]

# outer_padding = 30
# scene, layout = layoutscene(
#     outer_padding, 
#     resolution = (1200, 700),
#     backgroundcolor = RGBf0(0.98, 0.98, 0.98)
# )

# ax1 = layout[1, 1] = LAxis(scene, title="w_y=0.1, w_x varying")
# lines!(ax1, x_range, error_x)
# ax1.xlabel = "w_x"
# ax1.ylabel = "Loss MSE"

# ax2 = layout[1, 2] = LAxis(scene, title="w_x=0.0, w_y varying")
# lines!(ax2, y_range, error_y)
# ax1.xlabel = "w_y"
# ax1.ylabel = "Loss MSE"

# ax3 = layout[2, 1] = LAxis(scene, title="w_y=0.1, w_x varying, zoom")
# lines!(ax3, x_range[400:600], error_x[400:600])
# ax1.xlabel = "w_x"
# ax1.ylabel = "Loss MSE"

# ax4 = layout[2, 2] = LAxis(scene, title="w_x=0.1, w_y varying, zoom")
# lines!(ax4, y_range[400:600], error_y[400:600])
# ax1.xlabel = "w_y"
# ax1.ylabel = "Loss MSE"

# # what effect does the bias have?

# bias_range = LinRange(-1.0, 1.0, 2001)
# error_bias = [loss([0.0, 0.1, bias])[1] for bias in bias_range]

# bias_scene = lines(bias_range, error_bias)