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

function selkov(parameters, UA)
    a,b = parameters

    real_de = (du, u, p, t) -> begin
        du[1] = -u[1] + a*u[2] + (u[1].^2).*u[2]
        du[2] = b - a*u[2] - (u[1].^2)*u[2]
    end


    univ_de = (u, p, t) -> begin
        x,y = u
        z = UA(u,p)
        [-x + z[1] + (x^2)*y,
        b - a*y - (x^2)*y]
    end

    return (real_de, univ_de)
end

function selkov_sigma2(parameters, UA, α)
    a,b = parameters

    real_de = (du, u, p, t) -> begin
        du[1] = -u[1] + a*u[2] + (u[1].^2).*u[2]
        du[2] = b - a*u[2] - (u[1].^2)*u[2]
        du[3] = α*(u[1] - u[3])
        du[4] = α*(u[2] - u[4])

        # demean
        x_demeaned = u[1] - u[3]
        y_demeaned = u[2] - u[4]

        du[5] = α*(x_demeaned^2 - u[5])
        du[6] = α*(y_demeaned^2 - u[6])
    end


    univ_de = (u, p, t) -> begin
        x,y, xmean, ymean, xvar, yvar = u
        z = UA(u[1:2],p)
        dx = -x + z[1] + (x^2)*y
        dy = b - a*y - (x^2)*y
        dx_mean = α*(x - xmean)
        dy_mean = α*(y - ymean)
        x_demeaned = x - xmean
        y_demeaned = y - ymean
        dx_var = α*(x_demeaned^2 - xvar)
        dy_var = α*(y_demeaned^2 - yvar)
        return [dx, dy, dx_mean, dy_mean, dx_var, dy_var]
    end

    return (real_de, univ_de)
end

function ensemble_selkov(parameters, UA)
    a,b = parameters

    real_de = (du, u, p, t) -> begin
        du[1] = -u[1] + a*u[2] + (u[1].^2).*u[2]
        du[2] = b - a*u[2] - (u[1].^2)*u[2]
        du[3] = -u[3] + a*u[4] + (u[3].^2).*u[4]
        du[4] = b - a*u[4] - (u[3].^2)*u[4]
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
