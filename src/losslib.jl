function mse_loss(θ, y, predict)
    ŷ = predict(θ)
    loss = sum(abs2, y .- ŷ), ŷ
end