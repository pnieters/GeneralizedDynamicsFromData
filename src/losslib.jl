function mse_loss(θ, y, predict)
    ŷ = predict(θ)
    loss = sum(abs2, y .- ŷ), ŷ
end

function normed_ld_loss(θ, y, predict)
    y_differences = diff(y, dims=2)

    ŷ = predict(θ)
    ŷ_differences = diff(ŷ, dims=2)

    N = length(y_differences)

    loss = 1/N * sum([normed_ld(a,b) for (a,b) in zip(eachcol(y_differences), eachcol(ŷ_differences))])
    return loss, ŷ
end

function cosine_distance_loss(θ, y, predict)
    ŷ = predict(θ)
    N = length(y)
    loss = 1/N * sum([cosine_distance(a,b) for (a,b) in zip(eachcol(y), eachcol(ŷ))])
    return loss, ŷ
end

function combined_loss(θ, y, predict)
    y_differences = diff(y, dims=2)

    ŷ = predict(θ)
    ŷ_differences = diff(ŷ, dims=2)

    N = length(y)
    N_differences = length(y_differences)

    loss_length = 1/N * sum([normed_ld(a,b) for (a,b) in zip(eachcol(y_differences), eachcol(ŷ_differences))])
    loss_angle = 1/N * sum([cosine_distance(a,b) for (a,b) in zip(eachcol(y), eachcol(ŷ))])

    return loss_length + loss_angle, ŷ
end

normed_ld(a,b) = (norm(a)-norm(b))/(norm(a)+norm(b))

cosine_similarity(a, b) = dot(a,b) / (norm(a) * norm(b))
cosine_distance(a, b) = (1 - cosine_similarity(a, b))/2



# # Possible different norms/metrics
# cos_normed_ld(a,b) = (1 - cos((norm(a)-norm(b))/(norm(a)+norm(b)) * π)) / 2
# sqrt_cos_normed_ld(a,b) = sqrt(cos_normed_ld(a,b))
#
# angular_cosine_distance(a, b) = acos(cosine_similarity(a, b))/π
# sqrt_cos_distance(a,b) = sqrt(cosine_distance(a,b))