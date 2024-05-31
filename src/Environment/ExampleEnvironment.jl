module ExampleEnvironment

export prisonersdilemma, TransitionTensor, RewardTensor, actions, states, id

@kwdef mutable struct prisonersdilemma #Symmetric 2-agent 2-action Prisoners' Dilemma matrix game.
    reward::Float64  # reward of mutual cooperation
    temptation::Float64  # temptation of unilateral defection 
    suckerpayoff ::Float64  # sucker's payoff of unilateral cooperation
    punishment::Float64 # punishment of mutual defection
    N = 2
    M = 2
    Z = 1
    state = 0
end #of struct

function TransitionTensor(self::prisonersdilemma)
    Tsas = ones(self.Z, self.M, self.M,self.Z)
    return Tsas     
end # of function

function RewardTensor(self::prisonersdilemma)
    # """Get the Reward Tensor R[i,s,a1,...,aN,s']."""
    reward = zeros(2, self.Z, 2, 2, self.Z)
    reward[1, 1, :, :, 1] = [self.reward self.suckerpayoff; self.temptation self.punishment]
    reward[2, 1, :, :, 1] = [self.reward self.temptation; self.suckerpayoff self.punishment]

    return reward
end #of function

function actions(self::prisonersdilemma)
    return [['c' 'd'] for _ in 1:self.N]
end # of function

function states(self::prisonersdilemma)
    return ['.']    
end #of function

function id(self::prisonersdilemma)
    #returns id string of environment
    id = string(self.__class__.__name__, "_", self.temptation, "_", self.reward, "_", self.punishment, "_", self.suckerpayoff)    
end #of  function

end #of module