function mse_loss(θ, y, predict)
    ŷ = predict(θ)
    loss = sum(abs2, y .- ŷ), ŷ
end

function polar_loss(θ, y, w, predict)
    ŷ = predict(θ)
    N = size(ŷ, 2)
    loss = sum(w[1] * [normed_ld(y[:,i], ŷ[:,i]) for i in 1:N] +
               w[2] * [cosine_distance(y[:,i], ŷ[:,i]) for i in 1:N])
    return loss, ŷ
end

normed_ld(a,b) = abs(norm(a)-norm(b))/(norm(a)+norm(b))

cosine_similarity(a, b) = dot(a,b) / (norm(a) * norm(b))
cosine_distance(a, b) = (1 - cosine_similarity(a, b))/2