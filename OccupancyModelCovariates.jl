using Turing, DynamicHMC
using StatsFuns: logistic
using MCMCChains, StatsPlots
using Random

## simulate the data
Random.seed!(12)
S = 100  # number of sites
J = 5 # number of surveys in each site
detectionHistory = Array{Real}(undef, S * J)

## linear predictor for occupancy (ψ)
# continuous predictor
x1 = rand(Normal(0, 1), S)
x1_elongated = repeat(x1, J)
# matrix X1
X1 = Array{Float64}(undef, S, 2)
# populate matrix X1
for i in 1:S
        X1[i, :] = vcat(1, x1[i])
    end
# coeffcients vector (model parameters)
β1 = vcat(1, 3)
# probability of presence (occupancy)
ψ = logistic.(X1 * β1)

# occupancy data
z = Array{Bool}(undef, S, 1)
for i in 1:S
    z[i] = rand(Bernoulli(ψ[i]), i)[1]
end

z_elongated = repeat(z, J) # elongate z

## linear predictor for detection (p)
# continuous predictor
x2 = rand(Normal(0, 1), S)
x2_elongated = repeat(x2, J)
# matrix X2
X2 = Array{Float64}(undef, S, 2)
# populate matrix X2
for i in 1:S
        X2[i, :] = vcat(1, x2[i])
    end
# coeffcients vector (model parameters)
β2 = vcat(1, -3)
# probability of presence (occupancy)
p = logistic.(X2 * β2)
p_elongated = repeat(p, J)
# populate observed surveys data (detection history)
for s in 1:(S*J)
    detectionHistory[s] = rand(Bernoulli(z_elongated[s] * p_elongated[s]), 1)[1]
end


# occupancy model declaration
@model occupancy(occ1, occu2, detectionHist, x1, x2) = begin
    # priors
    intercept1 ~ Normal(0, 10)
    intercept2 ~ Normal(0, 10)
    β1 ~ Normal(0, 10)
    β2 ~ Normal(0, 10)
    # likelihood
    for i in 1:length(occ1)
        ψ = logistic(intercept1 + β1 * x1[i])
        occ1[i] ~ Bernoulli(ψ)
    end
        for j in 1:length(occu2)
        p = logistic(intercept2 + β2 * x2[j])
            detectionHist[j] ~ Bernoulli(occu2[j] * p)
            end
    end

# Start the No-U-Turn Sampler (NUTS)
chains = mapreduce(c -> sample(occupancy(z, z_elongated, detectionHistory, x1_elongated, x2_elongated), NUTS(0.95), 1000), chainscat, 1:3)
# summary
display(chains)
# plot parameters
gr(leg = false, bg = :black)
plot(chains, linecolor = [:white :green :orange])
