using GeneralizedDynamicsFromData
using Plots
using HDF5
using OrderedCollections
using Dates

using GLMakie # CairoMakie does VectorGraphics; but doesn't do 3D?
using AbstractPlotting
using AbstractPlotting.MakieLayout

using Base.Threads


function experiment()

    all_summaries = Dict()
    losses = [mse_loss, normed_ld_loss, cosine_distance_loss, combined_loss]
    # repetitions = 1
    initial_parameters = [InitializationLoader(1),InitializationLoader(2),InitializationLoader(3)]

    combinations = collect(Iterators.product(losses, initial_parameters))

    @threads for (loss, init_p) in combinations
    # @threads for loss in losses
    # for loss in losses

        repetitions = init_p.n_sets
        println("Exp with $(String(Symbol(loss)))")


        net_config = OrderedDict([
            :inputs => 2,
            :outputs => 1,
            :neurons => 32,
            :layers => 0,
            :non_lin => tanh,
            # :initialization => Flux.glorot_normal
            :initialization => init_p
        ]) 

        problem = Dict([
            :equation => selkov,
            :parameters => [0.1 0.6],
            :u0 => Float32[1.0, 1.0],
            :tE => 30.0f0,
            :tspan => (0.0f0, 30.0f0),
            :ts => 0.1,
            :solver => Vern7,
            :loss => loss
        ])

        identifier = String(Symbol(loss)) * "$(init_p.grp_id)"
        # identifier = String(Symbol(loss)) 

        println("<<< Starting Experiment for group $(init_p.grp_id) and Loss Function $(String(Symbol(loss)))")
        summary = repeat_experiment(problem, net_config, repetitions; progress=false)
        println("\t Concluded Group $(init_p.grp_id) | Loss $(String(Symbol(loss))) >>>")


        experiment_name = "selkov_$(identifier)"

        all_summaries[identifier] = summary

        h5open(experiment_name*".h5", "w") do file
            write(file, "losses", summary[:losses])
            write(file, "parameters", summary[:parameters])
            write(file, "predictions", summary[:predictions])
        end

    end

    return all_summaries
end

all_summaries = experiment()

# unpack
dict_grp_1_mse = all_summaries["mse_loss1"]
dict_grp_2_mse = all_summaries["mse_loss2"]
dict_grp_3_mse = all_summaries["mse_loss3"]

dict_grp_1_normed_ld = all_summaries["normed_ld_loss1"]
dict_grp_2_normed_ld = all_summaries["normed_ld_loss2"]
dict_grp_3_normed_ld = all_summaries["normed_ld_loss3"]

dict_grp_1_cosine_distance = all_summaries["cosine_distance_loss1"]
dict_grp_2_cosine_distance = all_summaries["cosine_distance_loss2"]
dict_grp_3_cosine_distance = all_summaries["cosine_distance_loss3"]

dict_grp_1_combined = all_summaries["combined_loss1"]
dict_grp_2_combined = all_summaries["combined_loss2"]
dict_grp_3_combined = all_summaries["combined_loss3"]

outer_padding = 30
scene, layout = layoutscene(
    outer_padding, 
    resolution = (700, 1600),
    backgroundcolor = RGBf0(0.98, 0.98, 0.98)
)

colors = [:red, :blue, :green]

###
ax_mse = layout[1, 1] = LAxis(scene, title="MSE Losses Sorted")
mse_sorted = [sort(dict_grp_1_mse[:losses]),
              sort(dict_grp_2_mse[:losses]),
              sort(dict_grp_3_mse[:losses])]
let first_el = 1
    for grp in 1:3
        L = length(mse_sorted[grp])
        last_el = (first_el - 1) + L

        # clean
        data = mse_sorted[grp]
        data[data .>= 5_000] .= NaN

        GLMakie.scatter!(ax_mse, collect(first_el:last_el), mse_sorted[grp], color = colors[grp])
        first_el += L
    end
end
ax_mse.xlabel = "Sorted ID in Groups"
ax_mse.ylabel = "MSE"

###
ax_nld = layout[2, 1] = LAxis(scene, title="Normed LD Losses Sorted")
normed_ld_sorted = [sort(dict_grp_1_normed_ld[:losses]),
                    sort(dict_grp_2_normed_ld[:losses]),
                    sort(dict_grp_3_normed_ld[:losses])]
let first_el = 1
    for grp in 1:3
        L = length(normed_ld_sorted[grp])
        last_el = (first_el - 1) + L
        GLMakie.scatter!(ax_nld, collect(first_el:last_el), normed_ld_sorted[grp], color = colors[grp])
        first_el += L
    end
end
ax_nld.xlabel = "Sorted ID in Groups"
ax_nld.ylabel = "Normed LD"

###
ax_cd = layout[3, 1] = LAxis(scene, title="Cosine Distance Losses Sorted")
cosine_distance_sorted = [sort(dict_grp_1_cosine_distance[:losses]),
                          sort(dict_grp_2_cosine_distance[:losses]),
                          sort(dict_grp_3_cosine_distance[:losses])]
let first_el = 1
    for grp in 1:3
        L = length(cosine_distance_sorted[grp])
        last_el = (first_el - 1) + L
        GLMakie.scatter!(ax_cd, collect(first_el:last_el), cosine_distance_sorted[grp], color = colors[grp])
        first_el += L
    end
end
ax_cd.xlabel = "Sorted ID in Groups"
ax_cd.ylabel = "Cosine Distance"

###
ax_combined = layout[4, 1] = LAxis(scene, title="Combined Cosine Distance + Normed LD Losses Sorted")
combined_sorted = [sort(dict_grp_1_combined[:losses]),
                   sort(dict_grp_2_combined[:losses]),
                   sort(dict_grp_3_combined[:losses])]
let first_el = 1
    for grp in 1:3
        L = length(combined_sorted[grp])
        last_el = (first_el - 1) + L
        GLMakie.scatter!(ax_combined, collect(first_el:last_el), combined_sorted[grp], color = colors[grp])
        first_el += L
    end
end
ax_combined.xlabel = "Sorted ID in Groups"
ax_combined.ylabel = "Cosine Distance"

save("Loss_Distributions_Sorted_and_Grouped_Cleaned.png", scene)