# Base

## Base class containing the core methods of MARLDynamics agents

module AgentsBase

using Revise
using UtilitiesHelpers

# The agent base class 
## The agent base class contains core methods to compute the strategy-average reward-prediction error. 
#### Das Paket enthält Kernmethoden, um den Fehler bei der Vorhersage der durchschnittlichen Belohnung für eine Strategie zu berechnen. Im Kontext von Multi-Agent Reinforcement Learning (MARL) deutet dies darauf hin, dass das Paket Funktionen oder Algorithmen bereitstellt, um den Fehler zwischen den erwarteten und tatsächlichen durchschnittlichen Belohnungen für bestimmte Strategien zu quantifizieren. Der Fehler in der Vorhersage kann eine wichtige Metrik sein, um die Leistung eines MARL-Algorithmus zu bewerten.

@kwdef mutable struct agentsbase
    TransitionTensor::AbstractArray{Float64,M} # transition model of the environment
    RewardTensor::AbstractArray{Float64,N}  # reward model of the environment
    DiscountFactors::Vector{Float64}  # the agents' discount factors
    use_prefactor::Bool = false # use the 1-DiscountFactor prefactor
    opteinsum::Bool = true # optimize einsum functions

    gamma = make_vaiable_vector(DiscountFactors, N)
    pre = if use_prefactor ? 1 - gamma : ones(N)
    
end #of struct

#default constrcutor
#agentsbase(TransitionTensor, RewardTensor, DiscountFactors;  use_prefactor = false, opteinsum = true) = abase(TransitionTensor, RewardTensor, DiscountFactors, use_prefactor, opteinsum)

fieldnames(agentsbase)


end #end of module