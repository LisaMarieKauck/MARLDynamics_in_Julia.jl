# Base

## Base class containing the core methods of MARLDynamics agents

module AgentsBase

    using Revise
    using UtilitiesHelpers

    # The agent base class 
    ## The agent base class contains core methods to compute the strategy-average reward-prediction error. 
    #### Das Paket enthält Kernmethoden, um den Fehler bei der Vorhersage der durchschnittlichen Belohnung für eine Strategie zu berechnen. Im Kontext von Multi-Agent Reinforcement Learning (MARL) deutet dies darauf hin, dass das Paket Funktionen oder Algorithmen bereitstellt, um den Fehler zwischen den erwarteten und tatsächlichen durchschnittlichen Belohnungen für bestimmte Strategien zu quantifizieren. Der Fehler in der Vorhersage kann eine wichtige Metrik sein, um die Leistung eines MARL-Algorithmus zu bewerten.

    Base.@kwdef mutable struct agentsbase
        TransitionTensor::AbstractArray{Float64} # transition model of the environment
        RewardTensor::AbstractArray{Float64}  # reward model of the environment
        DiscountFactors::Vector{Float64}  # the agents' discount factors
        use_prefactor::Bool = false # use the 1-DiscountFactor prefactor
        opteinsum::Bool = true # optimize einsum functions
        Omega::AbstractArray{Number}
        gamma::Vector{Number}
        N, M, Z, Q::Int 

        # number of agents
        N = size(RewardTensor)[1]  
        @assert length(size(TransitionTensor)[2:end-1]) == N "Inconsistent number of agents"
        @assert length(size(RewardTensor)[3:end-1]) == N "Inconsistent number of agents"

        # number of actions for each agent        
        M = size(TransitionTensor)[2] 
        @assert isapprox(size(TransitionTensor)[2:end-1], M) "Inconsisten number of actions"
        @assert isapprox(size(RewardTensor)[3:end-1], M) "Inconsisten number of actions"

        # number of states
        Z = size(TransitionTensor)[1] 
        Q = size(TransitionTensor)[1] 
        @assert size(TransitionTensor)[end-1] == Z "Inconsisten number of states"
        @assert size(RewardTensor)[end-1] == Z "Inconsisten number of states"
        @assert size(RewardTensor)[2] == Z "Inconsisten number of states"

        newagentsbase = new(RewardTensor, TransitionTensor, N, M, Z, Q)

        # discount factors
        gamma = make_variable_vector(DiscountFactors, N)

        # use (1-DiscountFactor) prefactor to have values on scale of rewards
        pre = if use_prefactor ? 1 - gamma : ones(N)

        # 'load' the other agents actions summation tensor for speed
        Omega = OtherAgentsActionsSummationTensor(N,M)
        has_last_statdist = false
        _last_statedist = ones(Z) / Z
          
    end #of struct
    #default constrcutor
    #agentsbase(TransitionTensor, RewardTensor, DiscountFactors;  use_prefactor = false, opteinsum = true) = abase(TransitionTensor, RewardTensor, DiscountFactors, use_prefactor, opteinsum)

    #fieldnames(agentsbase)

    function Tss(self::agentsbase, Xisa::AbstractArray)
        s = 1; # state s
        sprim = 2; # next state s'

        
    end

end #end of module
