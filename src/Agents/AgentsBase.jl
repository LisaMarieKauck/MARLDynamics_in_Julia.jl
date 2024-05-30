# Base

## Base class containing the core methods of MARLDynamics agents

module AgentsBase

using Revise
using UtilitiesHelpers


# The agent base class 
## The agent base class contains core methods to compute the strategy-average reward-prediction error. 
#### Das Paket enthält Kernmethoden, um den Fehler bei der Vorhersage der durchschnittlichen Belohnung für eine Strategie zu berechnen. Im Kontext von Multi-Agent Reinforcement Learning (MARL) deutet dies darauf hin, dass das Paket Funktionen oder Algorithmen bereitstellt, um den Fehler zwischen den erwarteten und tatsächlichen durchschnittlichen Belohnungen für bestimmte Strategien zu quantifizieren. Der Fehler in der Vorhersage kann eine wichtige Metrik sein, um die Leistung eines MARL-Algorithmus zu bewerten.


@kwdef mutable struct agentsbase
    TransitionTensor::AbstractArray{Float64} # transition model of the environment
    RewardTensor::AbstractArray{Float64}  # reward model of the environment
    DiscountFactors::Vector{Float64}  # the agents' discount factors
    use_prefactor::Bool = false # use the 1-DiscountFactor prefactor
    opteinsum::Bool = true # optimize einsum functions
    Omega::AbstractArray{Number}
    gamma::Vector{Number}
    N, M, Z, Q

    R=RewardTensor
    T=TransitionTensor

    # number of agents
    N = size(R)[1]  
    @assert length(size(T)[2:end-1]) == N, "Inconsistent number of agents"
    @assert length(size(R)[3:end-1]) == N, "Inconsistent number of agents"

    # number of actions for each agent        
    M = size(T)[2] 
    @assert isapprox(size(T)[2:end-1], M), "Inconsisten number of actions"
    @assert isapprox(size(R)[3:end-1], M), "Inconsisten number of actions"

    # number of states
    Z = size(T)[1] 
    Q = size(T)[1] 
    @assert size(T)[end-1] == Z, "Inconsisten number of states"
    @assert size(R)[end-1] == Z, "Inconsisten number of states"
    @assert size(R)[2] == Z, "Inconsisten number of states"

    newagentsbase = new(R, T, N, M, Z, Q)

    gamma = make_variable_vector(DiscountFactors, N)
    pre = if use_prefactor ? 1 - gamma : ones(N)

    function _OtherAgentsActionsSummationTensor(self::agentsbase)
        # to sum over the other agents and their respective actions using 'einsum'
        dim = vcat(self.N, # agent i 
                [self.N for _ in 1:(self.N-1)], #other agents
                self.M, # action a of agent i
                [self.M for _ in 1:self.N], # all acts
                [self.M for _ in 1:(self.N-1)]) #other actions
        Omega = zeros(Int64, Tuple(dim))

        for index in CartesianIndices(size(Omega))
            index = Tuple(index)
            I = index[1]
            notI = index[2 : self.N-1]
            A = index[self.N-1]
            allA = index[self.N : 2*self.N]
            notA = index[2*self.N:end]

            if length(unique(collect(Iterators.flatten((I, notI))))) == self.N
                # all agents indices are different

                if A == allA[I]
                    #action of agent i equals some other action
                    cd = allA[1:I] + allA[I+1:end] #other actions
                    areequal = [cd[k] == notA[k] for k in range(1,self.N-1)]
                    if all(areequal)
                        index = CartesianIndex(index)
                        Omega[index] = 1
                    end
                end
            end
        end #of for loop

    end #end of function
    
    
end #of struct

for index in CartesianIndices(size(Omega))
    index = Tuple(index)
    I = index[1] 
    notI = index[2 : N-1]
    print(notI, " next ") 
end

#default constrcutor
#agentsbase(TransitionTensor, RewardTensor, DiscountFactors;  use_prefactor = false, opteinsum = true) = abase(TransitionTensor, RewardTensor, DiscountFactors, use_prefactor, opteinsum)

fieldnames(agentsbase)


mutable struct MeinStruct
    p::Float64
    n::Int64
    f::Function
end

# Konstruktor für das Struct
function MeinStruct(; p::Float64 = 2.0, n::Int64 = 4, f::Function = x -> x + p)
    return MeinStruct(p, n, f)
end

# Verwendung des Structs
mein_objekt = MeinStruct(p=7)
ergebnis = mein_objekt.f(11)  # Ergebnis: 18



end #end of module

  