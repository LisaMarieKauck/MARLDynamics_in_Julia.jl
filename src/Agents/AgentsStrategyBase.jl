module AgentsStrategyActorCritic
# Base class containing the core methods of MARLDynamics agents in strategy space

using UtilitiesHelpers
using AgentsBase

struct strategybase <: agentsbase
    """
    Base class for deterministic strategy-average independent (multi-agent)
    temporal-difference reinforcement learning in strategy space.        
    """
    env # An environment object
    learning_rates::Union[float, Iterable] # agents' learning rates
    discount_factors::Union[float, Iterable] # agents' discount factors
    choice_intensities::Union[float, Iterable] # agents' choice intensities
    use_prefactor  # use the 1-DiscountFactor prefactor
    opteinsum  # optimize einsum 

end # of struct

function strategybase(env, 
    learning_rates, 
    discount_factors, 
    choice_intensities=1.0, 
    use_prefactor=false, 
    opteinsum=true)

    alpha = make_variable_vector(choice_intensities, N)


    return strategybase(env, learning_rates, discount_factors, choice_intensities, use_prefactor, opteinsum)
end # of function

function step(strategybase, 
    Xisa # Joint stragey
    )
    """
    Performs a learning step along the reward-prediction/temporal-difference error
    in strategy space, given joint strategy `Xisa`.
    """

    return Xisa
end # of function

end #of module