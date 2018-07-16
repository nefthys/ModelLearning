using Optim

using Distributions

# generate simple nonlinear dynamics
plant(x) = -x.^2

# Hyperparameters
K = 2  # number of Gaussians
d = 5  # dimension

batch_size = 100

# Generate some data
X = rand((batch_size, d))
V = plant(X)

function mixture(x, w, μ, Σ)
  normal_pdf(i) = pdf(MvNormal(μ[i, :], Σ[i, :]), x)
  return sum(weight*normal_pdf(i) for (i, weight) in enumerate(w))
end

function mse(G)
  w = G[:, 1]
  μ = G[:, 2:d+1]
  Σ = G[:, d+2:2*d+1]
 
  return sum(norm(V[i, :] - mixture(X[i, :], w, μ, Σ))^2 for i=1:batch_size)
end

# TODO non-unary callables in Optim?
# This may make some dumb assumptions
G = rand(K, 2*d+1)
optimize(mse, G)


# This doesn't work because nonlinear expressions don't allow vector data in JuMP :(
#=
using JuMP

model = Model()

@variable(model, W[1:K])  # Weights
@variable(model, μ[1:K, 1:d])  # Means
@variable(model, Σ[1:K, 1:d])  # Covariance diagonals

gaussian_pdf = @NLexpression(model, [i=1:batch_size, k=1:K],
                             exp(-(X[i, :] - μ[k, :])^2/(2*Σ[k, :]^2)/sqrt(2*π*Σ[k, :]^2)))

mixture = @NLexpression(model, [i=1:batch_size], sum([W[k]*(gaussian_pdf[i, k]) for k=1:K]))

@NLobjective(model, Min, sum([ norm(V[j, :] - mixture[j])^2 for j=1:batch_size]))
solve(model)
=#
