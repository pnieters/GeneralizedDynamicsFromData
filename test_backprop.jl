using GeneralizedDynamicsFromData
using Plots
using HDF5
using OrderedCollections
using Dates
using Flux

experiment_name = "test_backprop_$(Dates.month(today()))_$(Dates.day(today()))"
repetitions = 100

a, b = 0.1, 0.6
u0 = Float32[1.0, 1.0]

tE = 50.0f0
tspan = (0.0f0, 50.0f0)
ts = 0.1

w = [0.1, 0.0]
UA(x, _w) = _w[1] * x[1] + _w[2] * x[2]

params = Flux.params(w)

function real_de(du, u, p, t)
    du[1] = -u[1] + a*u[2] + (u[1].^2).*u[2]
    du[2] = b - a*u[2] - (u[1].^2)*u[2]
end

function univ_de(u, p, t)
    x,y = u
    z = UA(u,p)
    [-x + z[1] + (x^2)*y,
    b - a*y - (x^2)*y]
end

prob = ODEProblem(real_de, u0, tspan)
solution = solve(prob, Tsit5(), saveat=ts)

prob_nn = ODEProblem(univ_de, u0, tspan, w)

y = Array(solution)

predict(θ) = Array(concrete_solve(prob_nn, Tsit5(), u0, θ, saveat=solution.t))

guess = predict(w)
losses = [sum(abs2, y[:,i] .- guess[:,i]) for i in 1:size(y,2)]

function loss(ŷ)
    sum(abs2, y.-ŷ)
end

function gradient_i(i)
    gs = Flux.gradient(params) do
        ŷ = predict(w)
        L = sum(abs2, y[:,i] .- ŷ[:,i])
    end
end

grads = [gradient_i(i) for i in 1:size(y,2)]
grads = hcat([grads[i][w] for i in 1:length(grads)]...)

pre1 = plot(solution)
plot!(pre1, guess')
pre2 = plot(losses)
pre3 = plot(grads')
plot(pre1, pre2, pre3, layout=(3,1))

# redefine and cheat in 0 losses!
guess = predict(w)
y[:,42] .= guess[:,42]
y[:,84] .= guess[:, 84]
y[:,126] .= guess[:, 126]

losses = [sum(abs2, y[:,i] .- guess[:,i]) for i in 1:size(y,2)]

function loss(ŷ)
    sum(abs2, y.-ŷ)
end

function gradient_i(i)
    gs = Flux.gradient(params) do
        ŷ = predict(w)
        L = sum(abs2, y[:,i] .- ŷ[:,i])
    end
end

grads = [gradient_i(i) for i in 1:size(y,2)]
grads = hcat([grads[i][w] for i in 1:length(grads)]...)

post1 = plot(solution)
plot!(post1, guess')
post2 = plot(losses)
post3 = plot(grads')
plot(post1, post2, post3, layout=(3,1))