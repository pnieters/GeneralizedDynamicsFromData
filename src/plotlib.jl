function quantile_dict(data, quantiles)
    return Dict([(q, [quantile(d[:], q) for d in data]) for q in quantiles])
end

function weight_statistics(data)

    ymin, ymax = minimum(hcat(data...)), maximum(hcat(data...))
    x = range(1, stop=length(data), step=1)

    means = [mean(parameters) for parameters in data]

    quantiles = quantile_dict(data, [0.05, 0.25, 0.5, 0.75, 0.95])

    p = plot()

    plot!(p, x, means; linestyle=:solid, color=:black)
    plot!(p, x, quantiles[0.5]; 
          ribbon=(quantiles[0.5].-quantiles[0.25], quantiles[0.75].-quantiles[0.5]), 
          color=:blue, 
          fillalpha=.6)
    plot!(p, x, quantiles[0.5]; 
          ribbon=(quantiles[0.5].-quantiles[0.05], quantiles[0.95].-quantiles[0.5]),
          color=:blue,
          fillalpha=.3)

    plot!(p; xlims=(first(x), last(x)), ylims=(ymin, ymax), legend=false)

    return p
end

function weights_per_layer(data)

    n_layers = size(data[1],1)

    weights = [
        [data[t][l].weights for t in 1:length(data)] for l in 1:n_layers
    ]
    bias = [
        [data[t][l].bias for t in 1:length(data)] for l in 1:n_layers
    ]

    plots = []
    for layer in 1:n_layers
        push!(plots, weight_statistics(weights[layer]))
        push!(plots, weight_statistics(bias[layer]))
    end

    return plot(plots...; layout=(n_layers, 2))
end

function plot_weights(data, input_d, output_d, hidden_d, hidden_layers)
    all_parameters = weight_statistics(data)
    layer_parameters = weights_per_layer(reshape_all_parameters.(data, 
                                                                 input_d,
                                                                 output_d,
                                                                 hidden_d,
                                                                 hidden_layers))
    return plot(all_parameters, layer_parameters, layout=grid(2, 1, heights=[0.3, 0.7]))
end