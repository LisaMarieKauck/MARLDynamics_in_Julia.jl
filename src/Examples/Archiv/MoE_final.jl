using CSV
using DataFrames
using ReinforcementLearning
using Agents
using Random
using StaticArrays
using ReinforcementLearningBase

# Step 1: Load and preprocess data
data = CSV.read("Data/databrief-signatories.csv", DataFrame)
policies = CSV.read("Data/databrief-policies.csv", DataFrame)
function replace_spaces_in_columnnames!(df::DataFrame)
    for name in names(df)
        new_name = replace(name, " " => "_")
        new_name = replace(new_name, "." => "")
        new_name = replace(new_name, "-" => "_")
        rename!(df, name => new_name)
    end
    return df
end
replace_spaces_in_columnnames!(data)
replace_spaces_in_columnnames!(policies)

# Transform type of columns to Float/Int
data.BEI = parse.(Float64, replace.(data.BEI, "," => "."))
data.MEI = parse.(Float64, replace.(data.MEI, "," => "."))
data.Latitude = parse.(Float64, replace.(data.Latitude, "," => "."))
data.Longitude = parse.(Float64, replace.(data.Longitude, "," => "."))
data.Longitude = data.Longitude .+ abs(minimum(data.Longitude))

# Normalize relevant columns (GDP)
data[:, :NormalizedGDP] = data[:, :GDP] ./ maximum(data[:, :GDP])
data[:, :BEI_per_capita] = data[:, :BEI] ./ data[:, :BEI_INH]
data[:, :emission_red_per_capita] = data[:, :BEI] ./ data[:, :BEI_INH] - data[:, :MEI] ./ data[:, :MEI_INH]

# add summed up CO" reductions
completed_policies = filter(row -> row.Status_of_implementation == "COMPLETED", policies)
completed_policies = filter(row -> !ismissing(row.CO2_red), completed_policies)
co2_sums_compl = combine(groupby(completed_policies, :MUNI_CODE), :CO2_red => sum => :Total_CO2_red)
data = leftjoin(data, co2_sums_compl[:, ["MUNI_CODE","Total_CO2_red"]], on = "MUNI_CODE")
data[:, :Total_CO2_red] .= coalesce.(data[:, :Total_CO2_red], 0)
data[:, :Total_CO2_red] = Int64.(data[:, :Total_CO2_red]) 


# Define ClimateAction without type parameter
struct ClimateAction
    action::Symbol
end

# Step 2: Define Agent Structure
mutable struct CityAgent <: AbstractAgent
    id::Int
    pos::SVector{2, Float64}  
    muni_code::String
    state::Vector{Float64}
    actions::Vector{ClimateAction}  # Vector of ClimateAction without type parameter
    cumulative_reward::Float64
    q_table::Dict{Tuple{Vector{Float64}, ClimateAction}, Float64}  # Dict with ClimateAction as key
    learning_rate::Float64
    discount_factor::Float64
    co2_red_completed::Float64
end

# Step 3: Define environment
Base.@kwdef mutable struct ClimateEnv <: AbstractEnv
    agents::Vector{CityAgent}
    max_steps::Int
    step_counter::Int = 0
    rewards::Dict{Int, Float64} = Dict()
end

# Define available actions for each agent in the environment
function RLBase.action_space(env::ClimateEnv, agent::CityAgent)
    return agent.actions
end

## Reinforcement Learning Interface
function RLBase.reset!(env::ClimateEnv)
    env.step_counter = 0
    # Initialize rewards for each agent to ensure non-empty rewards dictionary
    env.rewards = Dict(agent.id => 0.0 for agent in env.agents)
end

# Define available actions for each agent in the environment
# Define action space for each agent in the environment
function RLBase.action_space(env::ClimateEnv, agent::CityAgent)
    return agent.actions  # Return the actions defined in the CityAgent instance
end

# Define action space for the whole environment
function RLBase.action_space(env::ClimateEnv)
    return [RLBase.action_space(env, agent) for agent in env.agents]  # List of action spaces per agent
end


# Discretize state space for Q-learning
function RLBase.state_space(env::ClimateEnv, ::Observation, ::DefaultPlayer)
    return [
        collect(0:0.2:1.0),           # Normalized GDP
        [0, 1e4, 1e5, 1e6, 1e7, 1e8], # Population
        [-15, 0, 10, 20, 30, 40, 50, 60, 80, 100], # emission_red_per_capita per capita
        [0, 50, 100, 150, 200],       # Policies aggregated
        [1, 2, 3],                    # Climate Code (3 intervals for Heating Degree Days)
        [0, 10, 20, 30, 40],          # N-Education Aggregated (sample bins)
        [0, 10, 20, 30, 40],          # N-Municipal self-governing aggregated
        [0, 10, 20, 30, 40],          # N-Financing Aggregated
        [0, 10, 20, 30, 40],          # N-Regulation Aggregated
        [0, 10, 20, 30, 40]           # N-Other Aggregated
    ]
end


function discretize_state(state::Vector{Float64}, state_space::Vector{Vector{Float64}})
    discretized_state = []
    for (i, value) in enumerate(state)
        bins = state_space[i]
        for j in 1:(length(bins)-1)
            if value < bins[j+1]
                push!(discretized_state, bins[j])
                break
            end
        end
    end
    return discretized_state
end

function RLBase.state(env::ClimateEnv, agent::CityAgent)
    return discretize_state(agent.state, RLBase.state_space(env, Observation(), DefaultPlayer()))
end

function RLBase.state(env::ClimateEnv, ::Observation, ::DefaultPlayer)
    # Return a list of states for each agent in the environment
    return [RLBase.state(env, agent) for agent in env.agents]
end


function RLBase.reward(env::ClimateEnv, agent::CityAgent)
    return get(env.rewards, agent.id, 0.0)
end

function RLBase.is_terminated(env::ClimateEnv)
    return env.step_counter >= env.max_steps
end

global_q_table = Dict{Tuple{Vector{Float64}, ClimateAction}, Float64}();

# Act! function without type parameter
function RLBase.act!(env::ClimateEnv, action::ClimateAction, agent::CityAgent)
    # Define reward based on action type
    individual_reward = action.action == :contribute ? -agent.co2_red_completed : 0
    
    # Calculate a collective reward based on all agents' actions
    if isempty(env.rewards)
        collective_reward = 0.0
    else
        collective_reward = sum(env.rewards[i] for i in keys(env.rewards)) / length(env.agents)
    end
    
    # Adjust the agent's reward to consider both individual and collective aspects
    reward = 0.5 * individual_reward + 0.5 * collective_reward  # weighted sum of both rewards
    
    env.rewards[agent.id] = get(env.rewards, agent.id, 0.0) + reward
    agent.cumulative_reward += reward

    # Q-learning update
    current_state = RLBase.state(env, agent)
    q_current = get(agent.q_table, (current_state, action), 0.0)
    max_q_next = maximum([get(agent.q_table, (current_state, a), 0.0) for a in agent.actions])
    td_target = reward + agent.discount_factor * max_q_next
    td_error = td_target - q_current

   # Update both the individual and global Q-tables
   agent.q_table[(current_state, action)] = q_current + agent.learning_rate * td_error
   global_q_table[(current_state, action)] = q_current + agent.learning_rate * td_error
end;

# Initialization of actions in `initialize_climate_env`
function initialize_climate_env(data::DataFrame, max_steps::Int)
    agents = []
    for (i, row) in enumerate(eachrow(data))
        state = [
            row[:NormalizedGDP],
            row[:N_Inhabit],
            row[:BEI_per_capita],
            row[:N_policies_aggregated],
            row[:Climate_Code],
            row[:N_Education_Aggregated],
            row[:N_Municipal_self_governing_aggregated],
            row[:N_Financing_Aggregated],
            row[:N_Regulation_Aggregated],
            row[:N_Other_Aggregated_]
        ]
        q_table = Dict{Tuple{Vector{Float64}, ClimateAction}, Float64}()

        # Define actions without type parameter
        actions = [ClimateAction(:contribute), ClimateAction(:defect)]

        agent = CityAgent(
            i,
            SVector(row[:Latitude], row[:Longitude]),
            "$(row[:MUNI_CODE])_$(row[:Country_Code])",
            state,
            actions,
            0.0,
            q_table,
            0.1,   # learning rate
            0.9,
            row[:Total_CO2_red]    # discount factor
        )
        push!(agents, agent)
    end
    return ClimateEnv(agents, max_steps, 0, Dict{Int, Float64}())
end


# Instantiate and run the environment
env = initialize_climate_env(data, 100)

function epsilon_greedy(agent::CityAgent, epsilon::Float64)
    if rand() < epsilon
        return rand(agent.actions)
    else
        state = discretize_state(agent.state, RLBase.state_space(env, Observation(), DefaultPlayer()))
        q_values = [get(global_q_table, (state, a), 0.0) for a in agent.actions]
        return agent.actions[argmax(q_values)]
    end
end;


function run_q_learning(env::ClimateEnv, num_episodes::Int, epsilon::Float64, epsilon_decay::Float64)
    for episode in 1:num_episodes
        RLBase.reset!(env)
        for step in 1:env.max_steps
            env.step_counter += 1 
            println("Step counter: ", env.step_counter)
            for agent in env.agents
                action = epsilon_greedy(agent, epsilon)
                RLBase.act!(env, action, agent)
                println("Episode $episode, Step $step, Agent $(agent.id) took action $(action.action), Reward $(RLBase.reward(env, agent))")
            end
            epsilon *= epsilon_decay
            if RLBase.is_terminated(env)
                break
            end
        end
    end
end;

run_q_learning(env, 500, 0.1, 0.99)

# Find the agent with the biggest reduction in cumulative reward
function find_agent_with_biggest_reduction(env::ClimateEnv)
    return findmin([agent.cumulative_reward for agent in env.agents])
end

min_reward, agent_index = find_agent_with_biggest_reduction(env);
best_agent = env.agents[agent_index];
println("The best agent is the agent with the id $agent_index: $best_agent with a reduction of $min_reward.")

function optimal_policy(agent::CityAgent)
    policy = Dict{Vector{Float64}, ClimateAction}()

    for (state, action) in keys(agent.q_table)
        q_value = agent.q_table[(state, action)]
        if !haskey(policy, state) || q_value > agent.q_table[(state, policy[state])]
            policy[state] = action
        end
    end

    return policy
end;

# Get the optimal policy for the best agent
best_agent_policy = optimal_policy(best_agent);
println("The optimal policy for the best agent was $best_agent_policy.")

# State Parameter Contribution with Multiple Contributions
function state_parameter_contribution(agent::CityAgent, state_space::Vector{Vector{Float64}})
    # Initialize a matrix to store contributions for each parameter and each action
    num_params = length(state_space)
    num_actions = length(agent.actions)
    contributions = zeros(num_params, num_actions)  # Rows: parameters, Columns: actions

    for (state, action) in keys(agent.q_table)
        base_q = agent.q_table[(state, action)]
        action_index = findfirst(==(action), agent.actions)
        for i in 1:min(length(state), num_params)  # Check length match
            # Perturb the ith parameter by moving it to the next bin if possible
            perturbed_state = copy(state)
            bins = state_space[i]

            # Check if we can perturb the parameter to the next bin
            bin_index = findfirst(==(perturbed_state[i]), bins)
            if bin_index !== nothing && bin_index < length(bins)
                # Perturb to the next bin value
                perturbed_state[i] = bins[bin_index + 1]
                perturbed_q = get(agent.q_table, (perturbed_state, action), base_q)
                contribution = abs(perturbed_q - base_q)
                contributions[i, action_index] += contribution
            end
        end
    end

    # Normalize contributions for each action to avoid NaN results
    for a in 1:num_actions
        total_contribution = sum(contributions[:, a])
        if total_contribution > 0
            contributions[:, a] /= total_contribution
        end
    end

    return contributions  # Returns a matrix with contributions per parameter per action
end

# Analyze which state parameter contributed the most for the best agent
println("The following state parameter contributed the most for the best agent:")
state_contributions = state_parameter_contribution(best_agent, RLBase.state_space(env, Observation(), DefaultPlayer()))
println(state_contributions)

# Function to Aggregate Contributions Across All Agents with Optimal Policies
function aggregate_contributions(env::ClimateEnv, state_space::Vector{Vector{Float64}})
    # Initialize a matrix to accumulate contributions for each parameter and each action
    num_params = length(state_space)
    num_actions = length(env.agents[1].actions)
    total_contributions = zeros(num_params, num_actions)
    agent_count = 0

    for agent in env.agents
        # Find the optimal policy for the agent
        optimal_policy_dict = optimal_policy(agent)
        
        # Calculate state parameter contributions for the agent
        agent_contributions = state_parameter_contribution(agent, state_space)
        
        # Accumulate contributions if the agent has an optimal policy
        if !isempty(optimal_policy_dict)
            total_contributions .+= agent_contributions
            agent_count += 1
        end
    end

    # Average the contributions if there were any agents with optimal policies
    if agent_count > 0
        return total_contributions / agent_count
    else
        println("No agents with an optimal policy found.")
        return total_contributions
    end
end

# Example of calling the function to get aggregated contributions
state_space = RLBase.state_space(env, Observation(), DefaultPlayer())
average_contributions = aggregate_contributions(env, state_space)
println("Average contributions of state parameters across agents following optimal policy:")
println(average_contributions)

# Function to identify agents whose optimal policy aligns with a specific action (=contribute)
function agents_with_optimal_action(env::ClimateEnv, state_space::Vector{Vector{Float64}}, target_action::Symbol)::Vector{CityAgent}
    matching_agents = CityAgent[]  # Ensures a vector of CityAgent type
    for agent in env.agents
        optimal_policy_dict = optimal_policy(agent)
        
        # Check if the majority of optimal actions align with the target action
        if !isempty(optimal_policy_dict)
            optimal_action_counts = count(x -> x == ClimateAction(target_action), values(optimal_policy_dict))
            if optimal_action_counts > length(optimal_policy_dict) / 2
                push!(matching_agents, agent)
            end
        end
    end
    return matching_agents
end

# Aggregates contributions across a list of agents
function aggregate_contributions_for_agents(agents::Vector{CityAgent}, state_space::Vector{Vector{Float64}})
    num_params = length(state_space)
    num_actions = length(agents[1].actions)
    total_contributions = zeros(num_params, num_actions)
    for agent in agents
        agent_contributions = state_parameter_contribution(agent, state_space)
        total_contributions .+= agent_contributions
    end
    return total_contributions / length(agents)
end

# Separate agents by optimal action type
agents_contribute = agents_with_optimal_action(env, state_space, :contribute)
agents_defect = agents_with_optimal_action(env, state_space, :defect)

# Check if both groups have agents before aggregating
if !isempty(agents_contribute) && !isempty(agents_defect)
    # Aggregate contributions for each group
    contribute_contributions = aggregate_contributions_for_agents(agents_contribute, state_space)
    defect_contributions = aggregate_contributions_for_agents(agents_defect, state_space)

    # Calculate and display the difference in contributions between groups
    difference_contributions = abs.(contribute_contributions - defect_contributions)
    println("Difference in contributions between agents with optimal policies of :contribute and :defect:")
    println(difference_contributions)
else
    println("One or both groups of agents are empty. Unable to compare contributions.")
end

# Function to find the 10 agents with the lowest cumulative rewards
function find_agents_with_minimal_reduction(env::ClimateEnv, num_agents::Int = 10)
    # Sort agents by cumulative reward in ascending order and take the first `num_agents`
    sorted_agents = sort(env.agents, by = agent -> agent.cumulative_reward)
    minimal_agents = sorted_agents[1:min(num_agents, length(sorted_agents))]
    
    # Return the minimal agents and their corresponding rewards
    return [(agent.id, agent.cumulative_reward) for agent in minimal_agents]
end

find_agents_with_minimal_reduction(env)