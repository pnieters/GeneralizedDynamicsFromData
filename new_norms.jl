using GeneralizedDynamicsFromData
using OrderedCollections
using Statistics
using LinearAlgebra

using GLMakie # CairoMakie does VectorGraphics; but doesn't do 3D?
using AbstractPlotting
using AbstractPlotting.MakieLayout

experiment_name = "new_norms"
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

normed_ld(a,b) = abs(norm(a)-norm(b))/(norm(a)+norm(b))
cos_normed_ld(a,b) = (1 - cos((norm(a)-norm(b))/(norm(a)+norm(b)) * π)) / 2
sqrt_cos_normed_ld(a,b) = sqrt(cos_normed_ld(a,b))

cosine_similarity(a, b) = dot(a,b) / (norm(a) * norm(b))
cosine_distance(a, b) = (1 - cosine_similarity(a, b))/2
angular_cosine_distance(a, b) = acos(clamp(cosine_similarity(a, b), -1, 1))/π
sqrt_cosine_distance(a,b) = sqrt(cosine_distance(a,b))

function bifurcation_cut(w_min, w_max, steps)

    # mse
    mse = Float64[]

    # length metrics
    nld = Float64[]
    cnld = Float64[]
    scnld = Float64[]

    # angular metrics
    cs = Float64[]
    cd = Float64[]
    acd = Float64[]
    scd = Float64[]

    for w in range(w_min, stop=w_max, length=steps)
        guess = predict([0.0, w, 0.0])
        N = length(guess)

        push!(mse, mse_loss([0.0, w, 0.0], y, predict)[1])

        push!(nld, 1/N * sum([normed_ld(a, b) for (a,b) in zip(eachcol(guess), eachcol(y))]))
        push!(cnld, 1/N * sum([cos_normed_ld(a, b) for (a,b) in zip(eachcol(guess), eachcol(y))]))
        push!(scnld, 1/N * sum([sqrt_cos_normed_ld(a, b) for (a,b) in zip(eachcol(guess), eachcol(y))]))

        push!(cs, 1/N * sum([cosine_similarity(a, b) for (a,b) in zip(eachcol(guess), eachcol(y))]))
        push!(cd, 1/N * sum([cosine_distance(a, b) for (a,b) in  zip(eachcol(guess), eachcol(y))]))
        push!(acd, 1/N * sum([angular_cosine_distance(a, b) for (a,b) in zip(eachcol(guess), eachcol(y))]))
        # push!(scd, sum([sqrt_cosine_distance(a, b) for (a,b) in zip(eachcol(guess), eachcol(y))]))
    end

    return (mse, nld, cnld, scnld, cs, cd, acd, scd)
end

(mse, nld, cnld, scnld, cs, cd, acd, scd) = bifurcation_cut(-0.5, 0.5, 1001)

x_range = range(-0.5, stop=0.5, length=1001)

outer_padding = 30
scene, layout = layoutscene(
    outer_padding, 
    resolution = (700, 1600),
    backgroundcolor = RGBf0(0.98, 0.98, 0.98)
)

ax1 = layout[1, 1] = LAxis(scene, title="MSE Loss, w_y varying")
mse_line = lines!(ax1, x_range, mse)
ax1.xlabel = "w_y"
ax1.ylabel = "Loss MSE"

ax2 = layout[2, 1] = LAxis(scene, title="Length Metrics, w_y varying")
nld_line = lines!(ax2, x_range, nld, color = :red)
cnld_line = lines!(ax2, x_range, cnld, color = :blue)
scnld_line = lines!(ax2, x_range, scnld, color = :green)
ax2.xlabel = "w_y"
ax2.ylabel = "Normalized Length Metric"

ax3 = layout[3, 1] = LAxis(scene, title="Cosine Similarity, w_y varying")
cs_line = lines!(ax3, x_range, cs)
ax3.xlabel = "w_y"
ax3.ylabel = "Cosine Similarity"

ax4 = layout[4, 1] = LAxis(scene, title="Angular Metrics, w_y varying")
cd_line = lines!(ax4, x_range, cd, color = :red)
acd_line = lines!(ax4, x_range, acd, color = :blue)
ax4.xlabel = "w_y"
ax4.ylabel = "Normalized Angular Metric"