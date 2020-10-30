function truscott_brindley(parameters, UA)
    a,b,c,d = parameters

    real_de = (du, u, p, t) -> begin
        du[1] = b*u[1].*(1-u[1]) - u[2].*((u[1].^2)./(a^2+u[1].^2))
        du[2] = d*u[2].*((u[1].^2)./(a^2+u[1].^2)) - c*d*u[2]
    end

    univ_de = (u, p, t) -> begin
        P,Z = u
        z = UA(u,p)
        [b*P*(1-P) + z[1],                 #z[1]=- Z*((P^2)/(p_[1]^2+P^2)),
         d*Z*((P^2)/(a^2+P^2)) + z[2]]  #z[2]=- p_[3]*p_[4]*Z]
    end

    return (real_de, univ_de)
end