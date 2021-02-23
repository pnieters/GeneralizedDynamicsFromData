function regression_model(input_d::Int, 
                          output_d::Int, 
                          hidden_d::Int, 
                          hidden_layers::Int, 
                          nonlinearity::Function, 
                          initialization)
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

function repeat_experiment(problem, 
                           net_config::OrderedDict, 
                           repetitions::Int; 
                           ε = 0.0, 
                           progress=true)

    callbacks = []
    min_losses = []
    min_params = []

    real_de, _ = problem[:equation](problem[:parameters], x->nothing)
    prob = ODEProblem(real_de, problem[:u0], problem[:tspan])
    solution = solve(prob, problem[:solver](), saveat=problem[:ts])
    y = Array(solution)

    Threads.@threads for _ in 1:repetitions

        UA = regression_model(values(net_config)...)
        _ , univ_de = problem[:equation](problem[:parameters], UA)

        θ₀ = initial_params(UA)
        prob_nn = ODEProblem(univ_de, problem[:u0], problem[:tspan], θ₀)

        ỹ = y .+ ε * randn(eltype(y), size(y))

        predict(θ) = Array(concrete_solve(prob_nn, problem[:solver](), problem[:u0], θ, saveat=solution.t))
        loss(θ) = problem[:loss](θ, y, predict)

        callback = CallbackLog(T=Float32)

        res = DiffEqFlux.sciml_train(loss, 
                                     θ₀, 
                                     problem[:optimizer](), 
                                     cb=callback, 
                                     maxiters=problem[:max_iter]; 
                                     progress)
        push!(callbacks, callback)
        push!(min_losses, res.minimum)
        push!(min_params, res.minimizer)
    end

    return Dict([
        :losses => min_losses,
        :parameters => min_params,
    ]), callbacks

end

function grid_experiment(problem, grid_config, repetitions; progress=true)

    result = []

    for net_config in grid_config
        summary = repeat_experiment(problem, net_config, repetitions; progress)
        push!(result, summary[:losses])
    end

    return reshape(result, size(grid_config))
end

# very purpose build data loader to repeat experiments with the same initializations
mutable struct InitializationLoader{T}
    _state::Int

    n_sets::Int
    n_parameters::Int
    grp_id::Int

    data_path::String
    initializations::Vector{Vector{T}}
end

function InitializationLoader(grp::Int)
    data_path = "./data/P$(grp)SelkovsimpleWeightsSolLoss.jld2"
    groupname = "P$(grp)G"

    all_parameters = load(data_path, groupname)
    initializations = [v[1] for v in all_parameters]
    n_sets = length(initializations)
    n_parameters = length(initializations[1])

    T = eltype(initializations[1])

    return InitializationLoader{T}(
        1,
        n_sets,
        n_parameters,
        grp,
        data_path,
        initializations
    )

end

function (il::InitializationLoader)()
    idx = il._state
    ret = il.initializations[idx]
    il._state = idx + 1
    ret
end

# ignore dimensions, in this experiment, we "know" everything.
function (il::InitializationLoader)(dims...)
    idx = il._state

    idx > il._n_sets || error("Too many iterations for number of init param sets")

    ret = il.initializations[idx]
    il._state = idx + 1
    ret
end

# # get a specific initialization
# function (il::InitializationLoader)(dims...; idx=1)
#     il.initializations[idx]
# end