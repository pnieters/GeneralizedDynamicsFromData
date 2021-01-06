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

function truscott_brindley1(parameters, UA)
    a,b,c,d = parameters

    real_de = (du, u, p, t) -> begin
        du[1] = b*u[1].*(1-u[1]) - u[2].*((u[1].^2)./(a^2+u[1].^2))
        du[2] = d*u[2].*((u[1].^2)./(a^2+u[1].^2)) - c*d*u[2]
    end

    univ_de = (u, p, t) -> begin
        P,Z = u
        z = UA(u,p)
        [b*P*(1-P) + z[1],                 #z[1]=- Z*((P^2)/(p_[1]^2+P^2)),
         d*Z*((P^2)/(a^2+P^2)) - c*d*Z]  #z[2]=- p_[3]*p_[4]*Z]
    end

    return (real_de, univ_de)
end

function selkov(parameters, UA)
    a,b = parameters

    real_de = (du, u, p, t) -> begin
        du[1] = -u[1] + a*u[2] + (u[1].^2).*u[2]
        du[2] = b - a*u[2] - (u[1].^2).*u[2]
    end


    univ_de = (u, p, t) -> begin
        x,y = u
        z = UA(u,p)
        [-x + z[1] + (x^2)*y,
        b - a*y - (x^2)*y]
    end

    return (real_de, univ_de)
end


function a_selkov(parameters, UA)
    a,b = parameters

    real_de = (du, u, p, t) -> begin
        du[1] = -u[1] + a*u[2] + (u[1].^2).*u[2]
        du[2] = b - a*u[2] - (u[1].^2).*u[2]
    end


    univ_de = (u, p, t) -> begin
        x,y = u
        z = UA(u,p)
        [-x + a*z[1] + (x^2)*y,
        b - a*y - (x^2)*y]
    end

    return (real_de, univ_de)
end

function ensemble_selkov(parameters, UA)
    a,b = parameters

    real_de = (du, u, p, t) -> begin
        du[1] = -u[1] + a*u[2] + (u[1].^2).*u[2]
        du[2] = b - a*u[2] - (u[1].^2).*u[2]
        du[3] = -u[3] + a*u[4] + (u[3].^2).*u[4]
        du[4] = b - a*u[4] - (u[3].^2).*u[4]
    end


    univ_de = (u, p, t) -> begin
        x,y,xx,yy = u
        z = UA(u,p)
        [-x + z[1] + (x^2)*y,
        b - a*y - (x^2)*y,
        -xx + z[2] + (xx^2)*yy,
        b - a*yy - (xx^2)*yy]
    end

    return (real_de, univ_de)
end

function ensemble_truscott_brindley(parameters, UA)
    a,b,c,d = parameters

    real_de = (du, u, p, t) -> begin
        du[1] = b*u[1].*(1-u[1]) - u[2].*((u[1].^2)./(a^2+u[1].^2))
        du[2] = d*u[2].*((u[1].^2)./(a^2+u[1].^2)) - c*d*u[2]
        du[3] = b*u[3].*(1-u[3]) - u[4].*((u[3].^2)./(a^2+u[3].^2))
        du[4] = d*u[4].*((u[3].^2)./(a^2+u[3].^2)) - c*d*u[4]
    end

    univ_de = (u, p, t) -> begin
        P,Z,PP,ZZ = u
        z = UA(u,p)
        [b*P*(1-P) - Z*((P^2)/(a^2+P^2)),                 #z[1]=- Z*((P^2)/(p_[1]^2+P^2)),
         d*Z*((P^2)/(a^2+P^2)) + z[1],
         b*PP*(1-PP) - ZZ*((PP^2)/(a^2+PP^2)),                 #z[1]=- Z*((P^2)/(p_[1]^2+P^2)),
          d*ZZ*((PP^2)/(a^2+PP^2)) + z[2]]  #z[2]=- p_[3]*p_[4]*Z]
    end

    return (real_de, univ_de)
end


function SaddleNode(parameters, UA)
    a = parameters

    real_de = (du, u, p, t) -> begin
        du[1] =  u[1]^2 + a
        du[2] = -u[2]
    end


    univ_de = (u, p, t) -> begin
        x,y = u
        z = UA(u,p)
        [z[1] + x^2,
        -y]
    end

    return (real_de, univ_de)
end


function transCritical(parameters, UA)
    a = parameters

    real_de = (du, u, p, t) -> begin
        du[1] = a*u[1] .- u[1].^2
        du[2] = -u[2]
    end


    univ_de = (u, p, t) -> begin
        x,y = u
        z = UA(u,p)
        [z[1] .- x.^2,
        -y]
    end

    return (real_de, univ_de)
end

function pitchfork(parameters, UA) # supercritical, subcritical du[1] = a*u[1] + u[1].^3
    a = parameters

    real_de = (du, u, p, t) -> begin
        du[1] = a.*u[1] .- u[1].^3
        du[2] = -u[2]
    end


    univ_de = (u, p, t) -> begin
        x,y = u
        z = UA(u,p)
        [z[1] - x.^3,
        -y]
    end

    return (real_de, univ_de)
end

# perioddoubling: Rossler attractor (http://scholarpedia.org/article/Rossler_attractor)

function Rossler(parameters, UA) # supercritical, subcritical du[1] = a*u[1] + u[1].^3
    a,b,c = parameters

    real_de = (du, u, p, t) -> begin
        du[1] = - u[2] .- u[3]
        du[2] = u[1] .+ a.*u[2]
        du[3] = b .+ u[3].*u[1] .- c.*u[3]
    end


    univ_de = (u, p, t) -> begin
        x,y,w = u
        z = UA(u,p)
        [-y .- w,
        x .+ z[1],
        b .+ w.*x .- c.*w]
    end

    return (real_de, univ_de)
end
