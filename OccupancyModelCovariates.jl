using Turing, DynamicHMC
using StatsFuns: logistic
using MCMCChains, StatsPlots
using Random

## simulate the data
Random.seed!(2345)
S = 100  # number of sites
J = 2 # number of surveys in each site
data = Array{Real}(undef, S, J) # declare data structure

## linear predictor for occupancy (ψ)
# continuous predictor
x1 = rand(Normal(0, 1), S)
# matrix X1
X1 = Array{Float64}(undef, S, 2)
# populate matrix X1
for i in 1:S
        X1[i, :] = vcat(1, x1[i])
    end
# coeffcients vector (model parameters)
β1 = vcat(0, 3)
# probability of presence (occupancy)
ψ = logistic.(X1 * β1)

# occupancy data
z = Array{Bool}(undef, S, 1)
for i in 1:S
    z[i] = rand(Bernoulli(ψ[i]), i)[1]
end
## linear predictor for detection (p)
# continuous predictor
x2 = rand(Normal(0, 0.5), S)
# matrix X2
X2 = Array{Float64}(undef, S, 2)
# populate matrix X2
for i in 1:S
        X2[i, :] = vcat(1, x2[i])
    end
# coeffcients vector (model parameters)
β2 = vcat(0, -4)
# probability of presence (occupancy)
p = logistic.(X2 * β2)

# populate observed surveys data (detection history)
for s in 1:S
    for j in 1:J
    data[s, j] = rand(Bernoulli(z[s]*p[s]), 1)[1]
    end
end

# occupancy model declaration
@model occupancy(S, J, data, z, x1, x2, σ1, σ2) = begin
    # priors
    intercept1 ~ Normal(0, 1)
    intercept2 ~ Normal(0, 1)
    β1 ~ Normal(0, σ1)
    β2 ~ Normal(0, σ2)
    # likelihood
    for s in S
        ψ = logistic(intercept1 + β1*x1[s])
        z[s] ~ Bernoulli(ψ)
        for j in J
        p = logistic(intercept2 + β2*x2[s])
            data[s, j] ~ Bernoulli(z[s]*p)
            return intercept1, intercept2, β1, β2
            end
        end
    end

# Start the No-U-Turn Sampler (NUTS)
chains = mapreduce(c -> sample(occupancy(S, J, data, z, x1, x2, 4, 7), NUTS(0.65), 1000), chainscat, 1:3)
# summary
display(chains)
# plot parameters
gr(leg = false, bg = :black)
plot(chains, linecolor = [:white :green :orange])
