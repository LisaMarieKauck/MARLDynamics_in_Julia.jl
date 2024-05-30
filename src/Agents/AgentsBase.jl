# Base

## Base class containing the core methods of MARLDynamics agents

module AgentsBase

    using Revise
    using UtilitiesHelpers

    # The agent base class 
    ## The agent base class contains core methods to compute the strategy-average reward-prediction error. 
    #### Das Paket enthält Kernmethoden, um den Fehler bei der Vorhersage der durchschnittlichen Belohnung für eine Strategie zu berechnen. Im Kontext von Multi-Agent Reinforcement Learning (MARL) deutet dies darauf hin, dass das Paket Funktionen oder Algorithmen bereitstellt, um den Fehler zwischen den erwarteten und tatsächlichen durchschnittlichen Belohnungen für bestimmte Strategien zu quantifizieren. Der Fehler in der Vorhersage kann eine wichtige Metrik sein, um die Leistung eines MARL-Algorithmus zu bewerten.

   mutable struct agentsbase
        TransitionTensor::AbstractArray{Float64} # transition model of the environment
        RewardTensor::AbstractArray{Float64}  # reward model of the environment
        DiscountFactors::Vector{Float64}  # the agents' discount factors
        use_prefactor::Bool # use the 1-DiscountFactor prefactor
        opteinsum::Bool # optimize einsum functions
        Omega::AbstractArray{Number}
        gamma::Vector{Number}
        N::Int
        M::Int
        Z::Int
        Q::Int
    end #of struct

    function agentsbase(TransitionTensor, # transition model of the environment
        RewardTensor,  # reward model of the environment
        DiscountFactors;  # the agents' discount factors
        use_prefactor=false,  # use the 1-DiscountFactor prefactor
        opteinsum=true)

        N = size(RewardTensor)[1] # number of agents
        @assert length(size(TransitionTensor)[2:end-1]) == N "Inconsistent number of agents"
        @assert length(size(RewardTensor)[3:end-1]) == N "Inconsistent number of agents"
        M = size(TransitionTensor)[2] # number of actions for each agent  
        @assert isapprox(size(TransitionTensor)[2:end-1], M) "Inconsisten number of actions"
        @assert isapprox(size(RewardTensor)[3:end-1], M) "Inconsisten number of actions"

        Z = size(TransitionTensor)[1]  # number of states
        Q = size(TransitionTensor)[1]  # number of states
        @assert size(TransitionTensor)[end-1] == Z "Inconsisten number of states"
        @assert size(RewardTensor)[end-1] == Z "Inconsisten number of states"
        @assert size(RewardTensor)[2] == Z "Inconsisten number of states"

        gamma = make_variable_vector(DiscountFactors, N)  # discount factors

        pre = use_prefactor ? 1 - gamma : ones(N) # use (1-DiscountFactor) prefactor to have values on scale of rewards
        # 'load' the other agents actions summation tensor for speed
        Omega = OtherAgentsActionsSummationTensor(N,M)
        has_last_statdist = false
        _last_statedist = ones(Z) / Z

        return agentsbase(TransitionTensor, RewardTensor, DiscountFactors, use_prefactor, opteinsum, Omega, gamma, N, M, Z, Q, pre, has_last_statdist, _last_statedist)
    end

    #fieldnames(agentsbase)

    function Tss(self::agentsbase, Xisa::AbstractArray)
        s = 1; # state s
        sprim = 2; # next state s'

    end

end #end of module


