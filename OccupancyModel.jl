using Turing
using MCMCChains, StatsPlots
using Random

# simulate the data
Random.seed!(12)
S = 150  # number of sites
J = 2 # number of surveys in each site
detectionHistory = Array{Real}(undef, S * J)

# true parameters
ψ = 0.8 # probability of presence (occupancy)
p = 0.5 # probability of detection

# occupancy data
z = rand(Bernoulli(ψ), S)
z_duplicated = [z z]
# populate observed surveys data (detections)
for s in 1:(S*J)
    detectionHistory[s] = rand(Bernoulli(z_duplicated[s]*p), 1)[1]
end

# occupancy model declaration
@model occupancy(d, occ1, occ2) = begin
    # priors
    ψ ~ Beta(1, 1)
    p ~ Beta(1, 1)
    # likelihood
    S = length(occ1)
    for i in 1:S
        occ1[i] ~ Bernoulli(ψ)
    end
    for j in 1:length(occ2)
        d[j] ~ Bernoulli(occ2[j] * p)
    end
end

# Start the No-U-Turn Sampler (NUTS)
chains = mapreduce(c -> sample(occupancy(detectionHistory, z, z_duplicated), NUTS(1000, .95), 1000, drop_warmup = false), chainscat, 1:3)
display(chains)

# trace plots and posteriors
plot(chains)

# running average plots
mp = meanplot(chains::Chains)

# plot joint density
post_ψ = chains[:ψ][:, 1]
post_p = chains[:p][:, 1]
jp = marginalkde(post_p, post_ψ)
plot(mp, jp, layout = (1, 2))
