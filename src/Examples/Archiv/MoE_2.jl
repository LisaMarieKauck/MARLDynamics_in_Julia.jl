using CSV
using DataFrames
using ReinforcementLearning
using Agents
using Random
using StaticArrays
using ReinforcementLearningBase

# Step 1: Load and preprocess data
data = CSV.read("Data/databrief-signatories.csv", DataFrame)
## Function to replace spaces with underscores in column names
function replace_spaces_in_columnnames!(df::DataFrame)
    for name in names(df)
        new_name = replace(name, " " => "_")
        new_name = replace(new_name, "." => "")
        rename!(df, name => new_name)
    end
    return df
end
replace_spaces_in_columnnames!(data)

## Transform type of columns to Float/Int
data.BEI = parse.(Float64, replace.(data.BEI, "," => "."))
data.MEI = parse.(Float64, replace.(data.MEI, "," => "."))
data.Latitude = parse.(Float64, replace.(data.Latitude, "," => "."))
data.Longitude = parse.(Float64, replace.(data.Longitude, "," => "."))
data.Longitude = data.Longitude .+ abs(minimum(data.Longitude))

## Normalize relevant columns (GDP)
data[:, :NormalizedGDP] = data[:, :GDP] ./ maximum(data[:, :GDP]);
data[:, :BEI_per_capita] = data[:, :BEI] ./ data[:, :BEI_INH];

## Define ClimateAction as a type for possible agent actions
struct ClimateAction{a}
    function ClimateAction(a)
        new{a}()
    end
end

# Step 2: Define Agent Structure
mutable struct CityAgent <: AbstractAgent
    id::Int
    pos::SVector{2, Float64}  
    muni_code::String
    state::Vector{Float64}  # Continuous state representation (e.g., economic data, emissions)
    actions::Vector{String}
    cumulative_reward::Float64
    q_table::Dict{Tuple{Vector{Float64}, ClimateAction{<:Any}}, Float64}  # Allow any ClimateAction
    learning_rate::Float64
    discount_factor::Float64
end

# Step 3: Define environment
Base.@kwdef mutable struct ClimateEnv <: AbstractEnv
    agents::Vector{CityAgent}
    max_steps::Int
    step_counter::Int = 0
    rewards::Dict{Int, Float64} = Dict()
end

## Reinforcement Learning Interface
function RLBase.reset!(env::ClimateEnv)
    env.step_counter = 0
    empty!(env.rewards)
end

# Define available actions for each agent in the environment
function RLBase.action_space(env::ClimateEnv, agent::CityAgent)
    return ClimateAction.([:contribute, :defect])
end

# Discretize state space for Q-learning
function RLBase.state_space(env::ClimateEnv, ::Observation, ::DefaultPlayer)
    return [
        collect(0:0.2:1.0),
        [0, 1e4, 1e5, 1e6, 1e7, 1e8],
        [0, 3e5, 6e5, 9e5, 1.2e6, 1.5e6, 1.7e6],
        [0, 50, 100, 150, 200]
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

function RLBase.reward(env::ClimateEnv, agent::CityAgent)
    return get(env.rewards, agent.id, 0.0)
end

function RLBase.is_terminated(env::ClimateEnv)
    return env.step_counter >= env.max_steps
end

function RLBase.act!(env::ClimateEnv, action::ClimateAction, agent::CityAgent)
    reward = action.action == :contribute ? -0.05 * rand() : -0.01 * rand()
    env.rewards[agent.id] = get(env.rewards, agent.id, 0.0) + reward
    agent.cumulative_reward += reward

    current_state = RLBase.state(env, agent)
    q_current = get(agent.q_table, (current_state, action), 0.0)

    max_q_next = maximum([get(agent.q_table, (current_state, a), 0.0) for a in agent.actions])
    td_target = reward + agent.discount_factor * max_q_next
    td_error = td_target - q_current

    agent.q_table[(current_state, action)] = q_current + agent.learning_rate * td_error
    env.step_counter += 1
end

function initialize_climate_env(data::DataFrame, max_steps::Int)
    agents = []
    for (i, row) in enumerate(eachrow(data))
        state = [
            row[:NormalizedGDP],
            row[:N_Inhabit],
            row[:BEI_INH],
            #row[:BEI_per_capita],
            row[:N_policies_aggregated]
        ]
        q_table = Dict{Tuple{Vector{Float64}, ClimateAction{<:Any}}, Float64}()

        # Create actions with specific type parameters
        actions = [ClimateAction(:contribute), ClimateAction(:defect)]
        
        agent = CityAgent(
            i, 
            SVector(row[:Latitude], row[:Longitude]), 
            string(row[:MUNI_CODE]), 
            state, 
            actions, 
            0.0, 
            q_table, 
            0.1,   # learning rate
            0.9    # discount factor
        )
        push!(agents, agent)
    end
    # Create the environment with a zero step counter and empty rewards dictionary
    return ClimateEnv(agents,  0, Dict{Int, Float64}())
end

env = initialize_climate_env(data, 5)

function epsilon_greedy(agent::CityAgent, epsilon::Float64)
    if rand() < epsilon
        return rand(agent.actions)
    else
        state = discretize_state(agent.state, RLBase.state_space(env, Observation(), DefaultPlayer()))
        q_values = [get(agent.q_table, (state, a), 0.0) for a in agent.actions]
        return agent.actions[argmax(q_values)]
    end
end


function run_q_learning(env::ClimateEnv, num_episodes::Int, epsilon::Float64, epsilon_decay::Float64)
    for episode in 1:num_episodes
        RLBase.reset!(env)
        for step in 1:env.max_steps
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
end

run_q_learning(env, 100, 0.1, 0.99)

