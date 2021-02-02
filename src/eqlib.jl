"""
    fritzhugh_nagumo(parameters, UA)

Return the FritzHugh-Nagumo[1,2] system of ordinary differential equations as an in-place 
function real_de!(du, u, p, t) to update the gradient vector du,
and a partical FritzHugh-Nagumo system as a universal differential equations[3], where
a non-linear term in the first equation and a linear term in the second equation must be 
learned from data (see below).

Parameters: Real[] with parameters [b0, b1, I, E] of the FritzHugh-Nagumo system
UA: Differentiable universal function approximator in the UDE sense: UA(u, p)

The FritzHugh-Nagumo system:
dx/dt = x - 1/3 x^3 - y + I
dy/dt = Eb₀ + Eb₁x - Ey

The partial FritzHugh-Nagumo system as a UDE:
dx/dt = x - 1/3 x^3 - y + I
dy/dt = Eb₀ + Uₚ([x, y]) - Ey

where UAₚ is the universal function approximator with weights p, and UA₁ is the first element
of the vector valued output.

We are interested in the system near a saddle-node bifurcation with
b0, b1, I, E = parameters = [0.9, 0.5, 1.2, 1.25] -> one stable state
b0, b1, I, E = parameters = [0.9, 0.5, 1.0, 1.25] -> two stable states, one unstable state

[1] FitzHugh, Richard. "Impulses and physiological states in theoretical models of nerve membrane." 
Biophysical journal 1.6 (1961): 445-466.
[2] Nagumo, Jinichi, Suguru Arimoto, and Shuji Yoshizawa. "An active pulse transmission line 
simulating nerve axon." Proceedings of the IRE 50.10 (1962): 2061-2070.
[3] Rackauckas, Christopher, et al. "Universal differential equations for scientific machine 
learning." arXiv preprint arXiv:2001.04385 (2020).
"""
function fritzhugh_nagumo(parameters, UA)
    b0, b1, I, E = parameters

    real_de! = (du, u, p, t) -> begin
        du[1] =  u[1] .- 1/3 .* u[1].^3 .- u[2] .+ I
        du[2] = E .*b0 .+ E .*b1 .*u[1] .- E .*u[2] 
    end


    univ_de = (u, p, t) -> begin
        x,y = u
        z = UA(u,p)
        [x .- 1/3 .* x.^3 .- y .+ I,
        E .*b0 .+ z[1] .- E .*y]
    end

    return (real_de!, univ_de)
end

"""
    genetic_toggle_switch(parameters, UA)

Return a genetic toggle switch (GTS) system of ordinary differential equations due to Gardner et al [1]
as an in-place function real_de!(du, u, p, t) to update the gradient vector du,
and a partical genetic toggle switch system as a universal differential equations[3], where
a non-linear term in the first equation and a linear term in the second equation must be 
learned from data (see below).

Parameters: Real[] with parameters [a1, a2, k, s] of the GTS system
UA: Differentiable universal function approximator in the UDE sense: UA(u, p)

The GTS system:
dx/dt = a₁/(1+yᵏ) - x
dy/dt = a₂/(1+xˢ) - y

The partial GTS system as a UDE:
dx/dt = a₁/(1+yᵏ) - x
dy/dt = a₂/(1+xˢ) - y

where UAₚ is the universal function approximator with weights p, and UA₁ is the first element
of the vector valued output.

We are interested in the system near a pitchfork bifurcation with, parameterization due to
Boshe and Ghosh[3]
a1, a2, k, s = parameters = [1.5, 1.5, 2.0, 2.0] -> one stable state
a1, a2, k, s = parameters = [3.5, 3.5, 2.0, 2.0] -> two stable states, one unstable state

[1] Gardner, Timothy S., Charles R. Cantor, and James J. Collins. "Construction of a genetic toggle 
switch in Escherichia coli." Nature 403.6767 (2000): 339-342.
[2] Rackauckas, Christopher, et al. "Universal differential equations for scientific machine 
learning." arXiv preprint arXiv:2001.04385 (2020).
[3] Bose, Indrani, and Sayantari Ghosh. "Bifurcation and criticality." Journal of Statistical 
Mechanics: Theory and Experiment 2019.4 (2019): 043403.
"""
function genetic_toggle_switch(parameters, UA) # gardner 2000 modell parametrisierung bose 2017
    a1, a2, k,s = parameters

    real_de = (du, u, p, t) -> begin
        du[1] = a1 ./(1 .+ u[2].^k) .- u[1]
        du[2] = a2 ./(1 .+ u[1].^s) .- u[2]
    end


    univ_de = (u, p, t) -> begin
        x,y = u
        z = UA(u,p)
        [z[1] .- x,
        a2 ./(1 .+ (x.^s)) .- y]
    end

    return (real_de, univ_de)
end

"""
    truscott_brindley(parameters, UA)

Return the Truscott-Brindley[1] system of ordinary differential equations as an in-place 
function real_de!(du, u, p, t) to update the gradient vector du,
and a partical Truscott-Brindley system as a universal differential equations[2], where
a non-linear term in the first equation and a linear term in the second equation must be 
learned from data (see below).

Parameters: Real[] with parameters [a, b, c, d] of the Truscott-Brindley system
UA: Differentiable universal function approximator in the UDE sense: UA(u, p)

The Truscott Brindley system:
dx/dt = bx(1-x) - y(x²/(a²+x²))
dy/dt = dy(x²/(a²+x²)) - cdy

The partial Truscott Brindley system as a UDE:
dx/dt = bx(1-x) + UAₚ([x, y])₁
dy/dt = dy(x²/(a²+x²)) + UAₚ([x,y])₂

where UAₚ is the universal function approximator with weights p, and UA₁ is the first element
of the vector valued output.

We are interested in the system near a Hopf bifurcation with
a, b, c, d = parameters = [0.053, 0.43, 0.024/(0.05*0.7), 0.05] -> oscillatory solution
a, b, c, d = parameters = [0.053, 0.43, 0.34, 0.05] -> steady state solution (Kirchfall)

[1] Truscott, J. E., and J. Brindley. "Ocean plankton populations as excitable media." Bulletin of 
Mathematical Biology 56.5 (1994): 981-998.
[2] Rackauckas, Christopher, et al. "Universal differential equations for scientific machine 
learning." arXiv preprint arXiv:2001.04385 (2020).
"""
function truscott_brindley(parameters, UA)
    a,b,c,d = parameters

    real_de! = (du, u, p, t) -> begin
        du[1] = b*u[1].*(1-u[1]) - u[2].*((u[1].^2)./(a^2+u[1].^2))
        du[2] = d*u[2].*((u[1].^2)./(a^2+u[1].^2)) - c*d*u[2]
    end

    univ_de = (u, p, t) -> begin
        P,Z = u
        z = UA(u,p)
        [b*P*(1-P) + z[1],                 #z[1]=- Z*((P^2)/(p_[1]^2+P^2)),
         d*Z*((P^2)/(a^2+P^2)) + z[2]]  #z[2]=- p_[3]*p_[4]*Z]
    end

    return (real_de!, univ_de)
end

"""
    roessler(parameters, UA)

Return the Truscott-Brindley[1] system of ordinary differential equations as an in-place 
function real_de!(du, u, p, t) to update the gradient vector du,
and a partical Truscott-Brindley system as a universal differential equations[2], where
a non-linear term in the first equation and a linear term in the second equation must be 
learned from data (see below).

Parameters: Real[] with parameters [a, b, c, d] of the Truscott-Brindley system
UA: Differentiable universal function approximator in the UDE sense: UA(u, p)

The Truscott Brindley system:
dx/dt = bx(1-x) - y(x²/(a²+x²))
dy/dt = dy(x²/(a²+x²)) - cdy

The partial Truscott Brindley system as a UDE:
dx/dt = bx(1-x) + UAₚ([x, y])₁
dy/dt = dy(x²/(a²+x²)) + UAₚ([x,y])₂

where UAₚ is the universal function approximator with weights p, and UA₁ is the first element
of the vector valued output.

We are interested in the system near a Hopf bifurcation with
a, b, c, d = parameters = [0.053, 0.43, 0.024/(0.05*0.7), 0.05] -> oscillatory solution
a, b, c, d = parameters = [0.053, 0.43, 0.34, 0.05] -> steady state solution (Kirchfall)

[1] Truscott, J. E., and J. Brindley. "Ocean plankton populations as excitable media." Bulletin of 
Mathematical Biology 56.5 (1994): 981-998.
[2] Rackauckas, Christopher, et al. "Universal differential equations for scientific machine 
learning." arXiv preprint arXiv:2001.04385 (2020).
"""
function roessler(parameters, UA) 
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