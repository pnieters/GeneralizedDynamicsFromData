function regression_model(input_d::Int, 
                          output_d::Int, 
                          hidden_d::Int, 
                          hidden_layers::Int, 
                          nonlinearity::Function, 
                          initialization::Function)
   if hidden_layers == 0
      return FastChain(FastDense(input_d, output_d))
   elseif hidden_layers == 1
      return FastChain(FastDense(input_d, hidden_d, nonlinearity; initW=initialization),
                       FastDense(hidden_d, output_d; initW=initialization))
   elseif hidden_layers > 1
      layers = [FastDense(input_d, hidden_d, nonlinearity; initW=initialization), 
                FastDense(hidden_d, output_d; initW=initialization)]
      for j in 1:hidden_layers-1
         insert!(layers, j+1, FastDense(hidden_d, hidden_d, nonlinearity; initW=initialization))
      end
      return FastChain(layers...)
   end

   error("Hidden layers must be a positive integer or 0")
end

function restructure_parameters(θ::Vector{T}, 
                                input_d::Int, 
                                output_d::Int, 
                                hidden_d::Int, 
                                hidden_layers::Int) where {T}
     """ Specialized on the type of regression neural networks we use here, and the FastX 
   chain of models """
   if hidden_layers == 0
      (output_d * input_d + output_d == length(θ)) || error("dimensions don't match")
      return [θ]
   elseif hidden_layers == 1
      l_1 = hidden_d * input_d + hidden_d
      l_N = output_d * hidden_d + output_d
      l_1 + l_N == length(θ) || error("dimensions don't match")
      return [θ[1:l_1], θ[l_1+1:l_1+l_N]]
   elseif hidden_layers > 1
      l_1 = hidden_d * input_d + hidden_d
      l_j = hidden_d * hidden_d + hidden_d
      l_N = output_d * hidden_d + output_d
      l_1 + (hidden_layers-1) * l_j + l_N == length(θ) || error("dimensions don't match")
      parameters = [θ[1:l_1], θ[end-l_N+1:end]]
      for j in 1:hidden_layers-1
         start = l_1 + (j-1) * l_j + 1
         stop = l_1 + j * l_j
         insert!(parameters, j+1, θ[start:stop])
      end
      return parameters
   end

   error("Hidden layers must be a positive integer or 0")
end

function reshape_all_parameters(θ::Vector{T},
                                input_d::Int,
                                output_d::Int,
                                hidden_d::Int,
                                hidden_layers::Int) where {T}
   _θ = restructure_parameters(θ, input_d, output_d, hidden_d, hidden_layers)
   reshape_all_parameters(_θ, input_d, output_d, hidden_d, hidden_layers)
end

function reshape_all_parameters(θ::Vector{Vector{T}},
                                input_d::Int,
                                output_d::Int,
                                hidden_d::Int,
                                hidden_layers::Int) where {T}
   length(θ) == hidden_layers + 1 || error("dimensions don't match: parameter length $(length(θ)) and expected $(hidden_layers+1) sets of parameters")

   if hidden_layers == 0
      return reshape_parameters(first(θ), input_d, output_d)
   elseif hidden_layers == 1
      return [reshape_parameters(first(θ), input_d, hidden_d),
              reshape_parameters(last(θ), hidden_d, output_d)]
   elseif hidden_layers > 1
      parameters = [reshape_parameters(first(θ), input_d, hidden_d),
                    reshape_parameters(last(θ), hidden_d, output_d)]
      for j in 2:hidden_layers
         insert!(parameters, j, reshape_parameters(θ[j], hidden_d, hidden_d))
      end
      return parameters
   end

end

function reshape_parameters(θ::Vector{T}, input_d::Int, output_d::Int) where {T}
   output_d * input_d + output_d == length(θ) || error("dimensions don't match")
   return (weights = reshape(θ[1:output_d*input_d], output_d, input_d), 
           bias = θ[output_d*input_d+1:end])
end

struct CallbackLog{T}
    log_params::Bool
    parameters::Vector{Vector{T}}

    log_loss::Bool
    losses::Vector{T}

    log_preds::Bool
    predictions::Vector{Array{T}}
end

function CallbackLog(;log_params=true, log_loss=true, log_preds=true, T=Any)
    CallbackLog(log_params,
                Vector{Vector{T}}(),
                log_loss,
                Vector{T}(),
                log_preds,
                Vector{Array{T}}())
end

function (cb::CallbackLog)(θ, loss, prediction)

    if cb.log_params
        push!(cb.parameters, copy(θ))
    end

    if cb.log_loss
        push!(cb.losses, loss)
    end
    
    if cb.log_preds
        push!(cb.predictions, prediction)
    end

    if any(isnan.(loss))
        println("NaN in losses; stop training!")
        return true
    elseif any(isinf.(loss))
        println("Inf in losses; stop training!")
        return true
    end

    return false
end

function repeat_experiment(problem, net_config::OrderedDict, repetitions)

    callbacks = []

    for _ in 1:repetitions

        UA = regression_model(values(net_config)...)
        real_de, univ_de = problem[:equation](problem[:parameters], UA)

        prob = ODEProblem(real_de, problem[:u0], problem[:tspan])
        solution = solve(prob, problem[:solver](), saveat=problem[:ts])

        θ₀ = initial_params(UA)

        prob_nn = ODEProblem(univ_de, problem[:u0], problem[:tspan], θ₀)
        # first_guess = concrete_solve(prob_nn, Vern7(), u0, θ₀, saveat=solution.t)

        y = Array(solution)
        predict(θ) = Array(concrete_solve(prob_nn, Vern7(), problem[:u0], θ, saveat=solution.t))
        loss(θ) = problem[:loss](θ, y, predict)

        callback = CallbackLog(T=Float32)

        res = DiffEqFlux.sciml_train(loss, θ₀, ADAM(1e-3), cb=callback, maxiters=200)
        push!(callbacks, callback)
    end

    return Dict([
        :losses => [c.losses[end] for c in callbacks],
        :parameters => hcat([c.parameters[end] for c in callbacks]...),
        :predictions => hcat([c.predictions[end] for c in callbacks]...)
    ])

end

function grid_experiment(problem, grid_config, repetitions)

    result = []

    for neuron_config in grid_config
        summary = repeat_experiment(problem, neuron_config, repetitions)
        push!(result, summary[:losses])
    end

    return reshape(result, size(grid_config))
end