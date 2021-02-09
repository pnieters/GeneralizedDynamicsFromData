function mse_loss(θ, y, predict)
    ŷ = predict(θ)
    loss = sum(abs2, y .- ŷ), ŷ
end

function mse_loss_norm(θ, y, predict)
    ŷ = predict(θ)
    N = size(y, 2)

    _loss = 1/N * sum([normed_mse(y[:,i], ŷ[:,i]) for i in 1:N])
    return _loss, ŷ 
end

normed_mse(a,b) = norm(a .- b)^2 / (0.001 + norm(a .- b)^2) # << neu

function normed_ld_loss(θ, y, predict)
    y_differences = diff(y, dims=2)

    ŷ = predict(θ)
    ŷ_differences = diff(ŷ, dims=2)

    N = size(y_differences, 2)

    _loss = 1/N * sum([normed_ld(y_differences[:,i], ŷ_differences[:,i]) for i in 1:N])
    return _loss, ŷ
end

function cosine_distance_loss(θ, y, predict)
    ŷ = predict(θ)
    N = size(y, 2)
    _loss = 1/N * sum([cosine_distance(y[:,i], ŷ[:,i]) for i in 1:N])
    return _loss, ŷ
end

function combined_loss(θ, y, predict)
    y_differences = diff(y, dims=2)

    ŷ = predict(θ)
    ŷ_differences = diff(ŷ, dims=2)

    N = size(y, 2)
    N_differences = size(y_differences, 2)

    loss_length = 1/N * sum([normed_ld(y_differences[:,i], ŷ_differences[:,i]) for i in 1:N_differences])
    loss_angle = 1/N * sum([cosine_distance(y[:,i], ŷ[:,i]) for i in 1:N])

    return loss_length + loss_angle, ŷ
end

function polar_loss(θ, y, w, predict)
    ŷ = predict(θ)
    N = size(ŷ, 2)
    loss = 1/N * sum(w[1] * [normed_ld(y[:,i], ŷ[:,i]) for i in 1:N] +
                     w[2] * [cosine_distance(y[:,i], ŷ[:,i]) for i in 1:N])
    return loss, ŷ
end

normed_ld(a,b) = abs(norm(a)-norm(b))/(norm(a)+norm(b))

cosine_similarity(a, b) = dot(a,b) / (norm(a) * norm(b))
cosine_distance(a, b) = (1 - cosine_similarity(a, b))/2



# # Possible different norms/metrics
# cos_normed_ld(a,b) = (1 - cos((norm(a)-norm(b))/(norm(a)+norm(b)) * π)) / 2
# sqrt_cos_normed_ld(a,b) = sqrt(cos_normed_ld(a,b))
#
# angular_cosine_distance(a, b) = acos(cosine_similarity(a, b))/π
# sqrt_cos_distance(a,b) = sqrt(cosine_distance(a,b))