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
    # Iterate over each column name, replacing " " with "_"
    for name in names(df)
        new_name = replace(name, " " => "_")
        new_name = replace(new_name, "." => "")
        rename!(df, name => new_name)
    end
    return df  # Return modified DataFrame
end
replace_spaces_in_columnnames!(data)

## Transform type of columns to Float/Int
names(data, String31);
data.BEI = parse.(Float64, replace.(data.BEI, "," => "."));
data.MEI = parse.(Float64, replace.(data.MEI, "," => "."));
data.Latitude = parse.(Float64, replace.(data.Latitude, "," => "."));
data.Longitude = parse.(Float64, replace.(data.Longitude, "," => "."));
data.Longitude = data.Longitude .+ abs(minimum(data.Longitude))
names(data, Int64);

## Normalize relevant columns (GDP)
data[:, :NormalizedGDP] = data[:, :GDP] ./ maximum(data[:, :GDP]);


# Step 2: Define Agent Structure
mutable struct CityAgent <: AbstractAgent
    id::Int
    pos::SVector{2, Float64}  
    muni_code::String
    state::Vector{Float64}  # Continuous state representation (e.g., economic data, emissions)
    actions::Vector{String}
    cumulative_reward::Float64
end

# Step 3: Define environment
Base.@kwdef mutable struct ClimateEnv <: AbstractEnv
    agents::Vector{CityAgent}          # List of agents in the environment
    max_steps::Int                     # Maximum steps before the episode terminates
    step_counter::Int = 0              # Step counter for each episode
    rewards::Dict{Int, Float64} = Dict()  # Dictionary to store rewards for each agent
end

## Define ClimateAction as a type for possible agent actions
struct ClimateAction{a}
    function ClimateAction(a)
        new{a}()
    end
end

## Reinforcement Learning Interface
function RLBase.reset!(env::ClimateEnv)
    env.step_counter = 0
    empty!(env.rewards)  # Clear any accumulated rewards
end

# Define available actions for each agent in the environment.
function RLBase.action_space(env::ClimateEnv, agent::CityAgent)
    return ClimateAction.([:contribute, :defect])  # Action space for each agent
end

function RLBase.action_space(env::ClimateEnv)
    return [RLBase.action_space(env, agent) for agent in env.agents]
end


# Return the state and state_space of an agent and for the environment
function RLBase.state(env::ClimateEnv, agent::CityAgent)
    return agent.state  # Return agent's state (e.g., economic data, emissions)
end

function RLBase.state(env::ClimateEnv, ::Observation, ::DefaultPlayer)
    # Return a list of states for each agent in the environment
    return [RLBase.state(env, agent) for agent in env.agents]
end

function RLBase.state_space(env::ClimateEnv, ::Observation, ::DefaultPlayer)
    return [
        (0.0, 1.0),            # Normalized GDP range
        (0.0, 1e8),            # Population (N_Inhabit) range
        (0.0, 1700000),        # BEI_INH range
        (0, 200)               # N_policies_aggregated range
    ]
end


# Define the reward function
function RLBase.reward(env::ClimateEnv, agent::CityAgent)
    return get(env.rewards, agent.id, 0.0)  # Get reward for agent from rewards dictionary
end

# Define the termination condition (is the episode over?)
function RLBase.is_terminated(env::ClimateEnv)
    return env.step_counter >= env.max_steps  # Terminate when max_steps is reached
end

# Step counter increment function
function RLBase.act!(env::ClimateEnv, action::ClimateAction, agent::CityAgent)
    reward = 0.0
    if action == ClimateAction(:contribute)
        reward = -0.05 * rand()  # Simulate a reward for "contribute"
    elseif action == ClimateAction(:defect)
        reward = -0.01 * rand()  # Simulate a reward for "defect"
    else
        @error "unknown action $action"
    end
    env.rewards[agent.id] = reward  # Store reward in the environment
    agent.cumulative_reward += reward  # Update agent's cumulative reward
    env.step_counter += 1
end

# Example of environment setup and running a one-shot game
function initialize_climate_env(data::DataFrame, max_steps::Int)
    # Initialize agents from data
    agents = []
    for (i, row) in enumerate(eachrow(data))
        state = [
            row[:NormalizedGDP],
            row[:N_Inhabit],
            row[:BEI_INH],
            row[:N_policies_aggregated]
        ]
        agent = CityAgent(i, SVector(row[:Latitude], row[:Longitude]), string(row[:MUNI_CODE]), state, ["contribute", "defect"], 0.0)
        push!(agents, agent)
    end

    # Create the environment with a zero step counter and empty rewards dictionary
    return ClimateEnv(agents, max_steps, 0, Dict{Int, Float64}())
end


# Instantiate environment and agents
env = initialize_climate_env(data, 5)

# Example action for an agent
for agent in env.agents
    action = RLBase.action_space(env, agent)[rand(1:2)]  # Randomly pick "contribute" or "defect"
    RLBase.act!(env, action, agent)
    println("Agent $(agent.id) took action $(action) and received reward $(RLBase.reward(env, agent))")
end