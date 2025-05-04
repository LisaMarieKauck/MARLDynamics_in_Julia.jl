# Base

## Base class containing the core methods of MARLDynamics agents

module AgentsBase

    using Revise
    using UtilitiesHelpers

    export agentsbase

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
        pre::Bool
        has_last_statdist::Bool
        _last_statedist::Vector
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

        return agentsbase(TransitionTensor, RewardTensor, DiscountFactors; use_prefactor, opteinsum)
    end

    #fieldnames(agentsbase)

    function Tss(self::agentsbase, Xisa::AbstractArray) # Average transition matrix
        s = 1; # state s
        sprim = 2; # next state s'
        b2d = collect(3 : (3+self.N-1)) #all actions
        X4einsum = reduce(vcat, (map(x -> [x[1], x[2]],zip(Xisa, [[s, b2d[a]] for a in 1:(self.N-1)])))) #pairs of state-action combinations
        #args = X4einsum + [self.TransitionTensor, [s]+b2d+[sprim], [s, sprim]]
        # This constructs a list args by concatenating X4einsum, self.T (the transition tensor), and two lists representing indices for the einsum function.
        #MISSING: EiNSUM of args, not clear how args should look like
    end #of function

    function Tisas(self::agentsbase,
        Xisa::AbstractArray  # Joint strategy
       )  #  Average transition Tisas)
        #"""Compute average transition model `Tisas`, given joint strategy `Xisa`"""      
        i = 0  # agent i
        a = 1  # its action a
        s = 2  # the current state
        s_ = 3  # the next state
        j2k = collect(4:(4+self.N-2))  # other agents
        b2d = collect((4+self.N-2):(4+self.N-2 + self.N-1))  # all actions
        e2f = collect((3+2*self.N-1):(3+2*self.N-1 + self.N-2))  # all other acts

        sumsis = [[j2k[l], s, e2f[l]] for l in (1:self.N-1)]  # sum inds
        otherX = reduce(vcat(map -> [x[1],x[2]], zip((self.N-2)*[Xisa], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f] + otherX\
            + [self.T, [s]+b2d+[s_], [i, s, a, s_]]
        return jnp.einsum(*args, optimize=self.opti)
    end #of function
       

      
    function Ris(self,
            Xisa:jnp.ndarray, # Joint strategy
            Risa:jnp.ndarray=None # Optional reward for speed-up
           ) -> jnp.ndarray: # Average reward
        """Compute average reward `Ris`, given joint strategy `Xisa`""" 
        if Risa is None  # for speed up
            # Variables      
            i = 0 
            s = 1
            sprim = 2
            b2d = list(range(3, 3+self.N))
        
            X4einsum = list(it.chain(*zip(Xisa,
                                    [[s, b2d[a]] for a in range(self.N)])))

            args = X4einsum + [self.T, [s]+b2d+[sprim],
                               self.R, [i, s]+b2d+[sprim], [i, s]]
            return jnp.einsum(*args, optimize=self.opti)
        
        else  # Compute Ris from Risa 
            i=0; s=1; a=2
            args = [Xisa, [i, s, a], Risa, [i, s, a], [i, s]]
            return jnp.einsum(*args, optimize=self.opti)
        end #of if
    end #of function
          
    function Risa(self,
             Xisa:jnp.ndarray # Joint strategy
            ) -> jnp.ndarray:  # Average reward
        """Compute average reward `Risa`, given joint strategy `Xisa`"""
        i = 0; a = 1; s = 2; s_ = 3  # Variables
        j2k = list(range(4, 4+self.N-1))  # other agents
        b2d = list(range(4+self.N-1, 4+self.N-1 + self.N))  # all actions
        e2f = list(range(3+2*self.N, 3+2*self.N + self.N-1))  # all other acts
 
        sumsis = [[j2k[l], s, e2f[l]] for l in range(self.N-1)]  # sum inds
        otherX = list(it.chain(*zip((self.N-1)*[Xisa], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f] + otherX\
            + [self.T, [s]+b2d+[s_], self.R, [i, s]+b2d+[s_],
               [i, s, a]]
        return jnp.einsum(*args, optimize=self.opti)  
    end #of function     
                 
    function Vis(self,
            Xisa:jnp.ndarray, # Joint strategy
            Ris:jnp.ndarray=None, # Optional reward for speed-up
            Tss:jnp.ndarray=None, # Optional transition for speed-up
            Risa:jnp.ndarray=None  # Optional reward for speed-up
           ) -> jnp.ndarray:  # Average state values
        """Compute average state values `Vis`, given joint strategy `Xisa`"""
        # For speed up
        Ris = self.Ris(Xisa, Risa=Risa) if Ris is None else Ris
        Tss = self.Tss(Xisa) if Tss is None else Tss
        
        i = 0  # agent i
        s = 1  # state s
        sp = 2  # next state s'

        n = np.newaxis
        Miss = np.eye(self.Z)[n,:,:] - self.gamma[:, n, n] * Tss[n,:,:]
        
        invMiss = jnp.linalg.inv(Miss)
               
        return self.pre[:,n] * jnp.einsum(invMiss, [i, s, sp], Ris, [i, sp],
                                          [i, s], optimize=self.opti)
    end #of function


    function Qisa(self,
             Xisa:jnp.ndarray, # Joint strategy
             Risa:jnp.ndarray=None, #  Optional reward for speed-up
             Vis:jnp.ndarray=None, # Optional values for speed-up
             Tisas:jnp.ndarray=None, # Optional transition for speed-up
            ) -> jnp.ndarray:  # Average state-action values
        """Compute average state-action values Qisa, given joint strategy `Xisa`"""
        # For speed up
        Risa = self.Risa(Xisa) if Risa is None else Risa
        Vis = self.Vis(Xisa, Risa=Risa) if Vis is None else Vis
        Tisas = self.Tisas(Xisa) if Tisas is None else Tisas

        nextQisa = jnp.einsum(Tisas, [0,1,2,3], Vis, [0,3], [0,1,2],
                              optimize=self.opti)

        n = np.newaxis
        return self.pre[:,n,n] * Risa + self.gamma[:,n,n]*nextQisa
    end #of function



end #end of module


