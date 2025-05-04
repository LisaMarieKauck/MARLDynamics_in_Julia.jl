using CSV
using DataFrames
using ReinforcementLearning
using Agents
using Random
using StaticArrays


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
mutable struct ClimateEnv <: AbstractEnv
    model::AgentBasedModel  # Embed the agent-based model
    max_steps::Int                      # Maximum steps in each episode
    step_counter::Int                   # Step counter for each episode
end

## Reinforcement Learning Interface
function ReinforcementLearningBase.reset!(env::ClimateEnv)
    env.step_counter = 0
    # Reset the agents' cumulative reward or state here
    for agent in env.model.agents
        agent.cumulative_reward = 0.0
    end
    return env  # Return the environment after resetting
end

# Define the observe function
function ReinforcementLearningBase.state(env::ClimateEnv)
    # Return the agent's state, which is the input to the policy
    return agent.state
end

# Define the action space for the agent
function ReinforcementLearningBase.action_space(env::ClimateEnv)
    return ["contribute", "defect"] 
end

# Define the reward function
function ReinforcementLearningBase.reward(env::ClimateEnv)
    return env.state == "contribute" ? -0.05 * rand() : -0.01 * rand()
end

# Define the termination condition (is the episode over?)
function ReinforcementLearningBase.is_terminated(env::ClimateEnv, agent::CityAgent)
    return env.step_counter >= env.max_steps
end

# Step counter increment function
function ReinforcementLearningBase.act!(env::ClimateEnv, action)
    env.step_counter += 1
    env.state += action 
    return env
end


# Step 5: Define the model initialization function
function initialize_model(data::DataFrame, episode_length::Int)
    # Define the continuous space for agents and the model
    space = ContinuousSpace((100, 100); spacing=4.0, periodic=false)
    model = AgentBasedModel(CityAgent, space)

     # Initialize Q-table as a dictionary for each agent
     Q_table = Dict{Int, Dict{Tuple{Vector{Float64}, String}, Float64}}()

    # Initialize agents and add them to the model
    for (i, row) in enumerate(eachrow(data))
        state = [
            row[:NormalizedGDP],
            row[:N_Inhabit],
            row[:BEI_INH],
            row[:N_policies_aggregated]
        ]
        # Define initial positions based on Latitude and Longitude
        pos = SVector(row[:Latitude], row[:Longitude]) # Convert latitude and longitude to appropriate spatial scale
        agent = CityAgent(i, pos, string(row[:MUNI_CODE]), state, ["contribute", "defect"], 0.0)
        add_agent!(agent, model)
        # Initialize Q-values for agent actions in each state
        Q_table[i] = Dict((state, action) => 0.0 for action in agent.actions)
    end

    # Create the reinforcement learning environment
    env = ClimateEnv(model, episode_length, 0)

    # Define the policy (epsilon-greedy with an epsilon of 0.1 for exploration)
    # policy = EpsilonGreedyPolicy(0.1)
    policy = (env, agent) -> ε_greedy_policy(Q_table, agent.state, agent.actions, 0.1)
    
    return env, policy, Q_table
end

## Policy Definition
function ε_greedy_policy(Q_table, state, actions, ε)
    if rand() < ε
        # Exploration: randomly select an action
        return rand(1:length(actions))
    else
        # Exploitation: choose the action with the highest Q-value
        action_values = [Q_table[(state, action)] for action in actions]
        return argmax(action_values)
    end
end


# Step 4: Define agent behavior (step function)
function agent_step!(agent::CityAgent, env::ClimateEnv, Q_table, α, γ)
    # Select an action based on the ε-greedy policy
    action_index = policy(env, agent)
    action = agent.actions[action_index]  # Map index to action name
    current_state = agent.state

    # Apply action and calculate reward
    if action == "contribute"
        reward = -0.05 * rand()  # Simulated reduction for contributing
    else
        reward = -0.01 * rand()  # Simulated lower reduction for defecting
    end

    # Update agent's cumulative reward
    agent.cumulative_reward += reward
    
    # Observe the new state (for simplicity, let's assume no state change)
    new_state = agent.state
    
    # Get the maximum Q-value for the new state
    max_future_q = maximum(get(Q_table[agent.id], (new_state, a), 0.0) for a in agent.actions)
    
    # Update the Q-value for the current state-action pair
    Q_table[agent.id][(current_state, action)] += α * (reward + γ * max_future_q - Q_table[agent.id][(current_state, action)])
    
    println("Agent $(agent.id) took action $action and received reward $reward")
end


# Step 5: Training with Results Logging
function train_and_log!(env::ClimateEnv, policy, Q_table, num_episodes::Int, α::Float64, γ::Float64)
    results = []  # Collect results here
    
    for episode in 1:num_episodes
        reset!(env)  # Reset environment for each episode
        total_reward = 0.0  # Track total reward for the episode
        
        while !is_terminated(env, first(env.model.agents))  # Check termination
            # Step through each agent in the environment
            for agent in env.model.agents
                agent_step!(agent, env, Q_table, α, γ)  # Execute agent step with the Q-learning update
                total_reward += agent.cumulative_reward  # Accumulate reward
            end
            
            step!(env)  # Increment the environment step counter
        end
        
        # Log results of the episode
        push!(results, total_reward)
        println("Episode $episode completed with total reward: $total_reward")
    end
    
    return results
end



# Main execution to initialize model and policy and run training
env, policy, Q_table = initialize_model(data, 20)
results = train_and_log!(env, policy, Q_table, 10, 0.1, 0.9)  # Train with α=0.1 and γ=0.9
# Train for 10 episodes and log results

# Display the results after training
println("Results of the training:")
println(results)
