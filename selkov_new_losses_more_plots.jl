using GeneralizedDynamicsFromData
using HDF5
using OrderedCollections
using Dates
using Statistics

using GLMakie # CairoMakie does VectorGraphics; but doesn't do 3D?
using AbstractPlotting
using AbstractPlotting.MakieLayout


function get_solution()
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
    tspan = (0.0f0, 30.0f0)
    ts = 0.1

    real_de, univ_de = selkov(parameters, UA)

    prob = ODEProblem(real_de, u0, tspan)
    solution = solve(prob, Vern7(), saveat=ts)

    y = Array(solution)
    return y
end

function read_results(loss)

    grps = ["1", "2", "3"]

    losses = Float32[]
    parameters = Array{Float32}(undef, 3, 0)
    predictions = Array{Float32}(undef, 2, 301, 0)

    for grp in grps
        filename = "selkov_"*loss*"_loss"*grp*".h5"
        _losses, _parameters, _predictions = h5open(filename, "r") do file
            read(file, "losses"), read(file, "parameters"), read(file, "predictions")
        end

        N = length(_losses)
        losses = vcat(losses, _losses)
        parameters = hcat(parameters, _parameters)
        predictions = cat(predictions, reshape(_predictions, 2, 301, N), dims=3)
    end

    return losses, parameters, predictions

end

function argmed(itr)
    sorted = sort(itr)
    # bias to the smaller element
    median_element = sorted[floor(Int, length(itr)/2)]
    findfirst(x->x==median_element, itr)   
end

# GLOBALS
outer_padding = 30
trange = collect(0.0:0.1:30.0)
y = get_solution()

mse_loss, mse_params, mse_preds = read_results("mse")
mse_loss[isnan.(mse_loss)] .= typemax(eltype(mse_loss))

best_mse_id = argmin(mse_loss)
best_mse_pred = mse_preds[:, :, best_mse_id]

med_mse_id = argmed(mse_loss)
med_mse_pred = mse_preds[:, :, med_mse_id]

mse_scene, layout_mse = layoutscene(
    outer_padding, 
    resolution = (700, 800),
    backgroundcolor = RGBf0(0.98, 0.98, 0.98)
)
ax_best = layout_mse[1,1] = LAxis(mse_scene, title="Best Solution: $(mse_loss[best_mse_id])")
lines!(ax_best, trange, y[1,:], color=:Red, linestyle=:dash)
lines!(ax_best, trange, y[2,:], color=:Blue, linestyle=:dash)

lines!(ax_best, trange, best_mse_pred[1,:], color=:Red)
lines!(ax_best, trange, best_mse_pred[2,:], color=:Blue)

ax_med = layout_mse[2,1] = LAxis(mse_scene, title="Median Solution: $(mse_loss[med_mse_id])")
lines!(ax_med, trange, y[1,:], color=:Red, linestyle=:dash)
lines!(ax_med, trange, y[2,:], color=:Blue, linestyle=:dash)

lines!(ax_med, trange, med_mse_pred[1,:], color=:Red)
lines!(ax_med, trange, med_mse_pred[2,:], color=:Blue)



normed_ld_loss, normed_ld_params, normed_ld_preds = read_results("normed_ld")
normed_ld_loss[isnan.(normed_ld_loss)] .= typemax(eltype(normed_ld_loss))

best_normed_ld_id = argmin(normed_ld_loss)
best_normed_ld_pred = normed_ld_preds[:, :, best_normed_ld_id]

med_normed_ld_id = argmed(normed_ld_loss)
med_normed_ld_pred = normed_ld_preds[:, :, med_normed_ld_id]

normed_ld_scene, layout_normed_ld = layoutscene(
    outer_padding, 
    resolution = (700, 800),
    backgroundcolor = RGBf0(0.98, 0.98, 0.98)
)
ax_best = layout_normed_ld[1,1] = LAxis(normed_ld_scene, title="Best Solution: $(normed_ld_loss[best_normed_ld_id])")
lines!(ax_best, trange, y[1,:], color=:Red, linestyle=:dash)
lines!(ax_best, trange, y[2,:], color=:Blue, linestyle=:dash)

lines!(ax_best, trange, best_normed_ld_pred[1,:], color=:Red)
lines!(ax_best, trange, best_normed_ld_pred[2,:], color=:Blue)

ax_med = layout_normed_ld[2,1] = LAxis(normed_ld_scene, title="Median Solution: $(normed_ld_loss[med_normed_ld_id])")
lines!(ax_med, trange, y[1,:], color=:Red, linestyle=:dash)
lines!(ax_med, trange, y[2,:], color=:Blue, linestyle=:dash)

lines!(ax_med, trange, med_normed_ld_pred[1,:], color=:Red)
lines!(ax_med, trange, med_normed_ld_pred[2,:], color=:Blue)


cosine_distance_loss, cosine_distance_params, cosine_distance_preds = read_results("cosine_distance")
cosine_distance_loss[isnan.(cosine_distance_loss)] .= typemax(eltype(cosine_distance_loss))

best_cosine_distance_id = argmin(cosine_distance_loss)
best_cosine_distance_pred = cosine_distance_preds[:, :, best_cosine_distance_id]

med_cosine_distance_id = argmed(cosine_distance_loss)
med_cosine_distance_pred = cosine_distance_preds[:, :, med_cosine_distance_id]

cosine_distance_scene, layout_cosine_distance = layoutscene(
    outer_padding, 
    resolution = (700, 800),
    backgroundcolor = RGBf0(0.98, 0.98, 0.98)
)
ax_best = layout_cosine_distance[1,1] = LAxis(cosine_distance_scene, title="Best Solution: $(cosine_distance_loss[best_cosine_distance_id])")
lines!(ax_best, trange, y[1,:], color=:Red, linestyle=:dash)
lines!(ax_best, trange, y[2,:], color=:Blue, linestyle=:dash)

lines!(ax_best, trange, best_cosine_distance_pred[1,:], color=:Red)
lines!(ax_best, trange, best_cosine_distance_pred[2,:], color=:Blue)

ax_med = layout_cosine_distance[2,1] = LAxis(cosine_distance_scene, title="Median Solution: $(cosine_distance_loss[med_cosine_distance_id])")
lines!(ax_med, trange, y[1,:], color=:Red, linestyle=:dash)
lines!(ax_med, trange, y[2,:], color=:Blue, linestyle=:dash)

lines!(ax_med, trange, med_cosine_distance_pred[1,:], color=:Red)
lines!(ax_med, trange, med_cosine_distance_pred[2,:], color=:Blue)


combined_loss, combined_params, combined_preds = read_results("combined")
combined_loss[isnan.(combined_loss)] .= typemax(eltype(combined_loss))

best_combined_id = argmin(combined_loss)
best_combined_pred = combined_preds[:, :, best_combined_id]

med_combined_id = argmed(combined_loss)
med_combined_pred = combined_preds[:, :, med_combined_id]

combined_scene, layout_combined = layoutscene(
    outer_padding, 
    resolution = (700, 800),
    backgroundcolor = RGBf0(0.98, 0.98, 0.98)
)
ax_best = layout_combined[1,1] = LAxis(combined_scene, title="Best Solution: $(combined_loss[best_combined_id])")
lines!(ax_best, trange, y[1,:], color=:Red, linestyle=:dash)
lines!(ax_best, trange, y[2,:], color=:Blue, linestyle=:dash)

lines!(ax_best, trange, best_combined_pred[1,:], color=:Red)
lines!(ax_best, trange, best_combined_pred[2,:], color=:Blue)

ax_med = layout_combined[2,1] = LAxis(combined_scene, title="Median Solution: $(combined_loss[med_combined_id])")
lines!(ax_med, trange, y[1,:], color=:Red, linestyle=:dash)
lines!(ax_med, trange, y[2,:], color=:Blue, linestyle=:dash)

lines!(ax_med, trange, med_combined_pred[1,:], color=:Red)
lines!(ax_med, trange, med_combined_pred[2,:], color=:Blue)
