using Turing
using MCMCChains, StatsPlots
using Random

# simulate the data
Random.seed!(24)
S = 50  # number of sites
J = 2 # number of surveys in each site
data = Array{Real}(undef, S, J)

# true parameters
ψ = 0.8 # probability of presence (occupancy)
p = 0.3 # probability of detection

# occupancy data
z = rand(Bernoulli(ψ), S)

# populate observed surveys data (detections)
for s in 1:S
    for j in 1:J
    data[s, j] = rand(Bernoulli(z[s]*p), 1)[1]
    end
end

# occupancy model declaration
@model occupancy(data, z) = begin
    # priors
    ψ ~ Beta(2, 2)
    p ~ Beta(2, 2)
    # likelihood
    for s in S
        z[s] ~ Bernoulli(ψ)
        for j in J
            data[s, j] ~ Bernoulli(z[s]*p)
            return ψ, p
            end
        end
    end

# Start the No-U-Turn Sampler (NUTS)
chains = mapreduce(c -> sample(occupancy(data, z), NUTS(0.65), 1000), chainscat, 1:3)
display(chains)

# trace plots and posteriors
plot(chains)

# running average plots
mp = meanplot(chains::Chains)

# plot joint posterior distribution
post_ψ = chains[:ψ][:, 1]
post_p = chains[:p][:, 1]
jp = marginalkde(post_p, post_ψ)
plot(mp, jp, layout = (1, 2))
