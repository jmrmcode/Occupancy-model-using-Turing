# Occupancy-model-using-Turing
MacKenzie et al's (2002) species occupancy model using Turing
## Purpose
Species occurrence (or "site occupancy") is an important variable when researching wildlife. Let'say *Z*<sub>i</sub> is a binary variable taking *Z*<sub>i</sub> = 1 when site *i* is occupied by a species and *Z*<sub>i</sub> = 0 otherwise. False absences, i.e., a species is not detected but present, are frequent when collecting field data. The detection probability *p* is defined as the probability of detecting a species when present at site *i* (*Z*<sub>i</sub> = 1). MacKenzie et al's (2002) proposed the following hierarchical modeling approach to model species occupancy while accounting for imperfect detection. 

*Z*<sub>i</sub> ~ Bernoulli(&Psi;) where *i* = 1, 2, ..., *S*    sites   [1]

Y<sub>ij</sub>|*Z*<sub>i</sub> ~ Bernoulli(*Z*<sub>i</sub> * *p*)    for *j* = 1, 2, 3, ..., *J*<sub>i</sub> surveys  [2]

Note that Eq[1] describes the latent occupancy state variable and Eq[2] is the joint distribution of the observations conditional on the latent occupancy state. 
