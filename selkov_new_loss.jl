using GeneralizedDynamicsFromData
using Plots
using HDF5
using OrderedCollections
using Dates

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
            :tE => 50.0f0,
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