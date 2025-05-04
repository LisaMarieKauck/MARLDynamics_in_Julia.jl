using CSV
using DataFrames
using ReinforcementLearning
using Agents

# Step 1: Load and preprocess data
data = CSV.read("Data/databrief-signatories.csv", DataFrame)
## Function to replace spaces with underscores in column names
function replace_spaces_in_columnnames!(df::DataFrame)
    # Iterate over each column name, replacing " " with "_"
    for name in names(df)
        new_name = replace(name, " " => "_")
        rename!(df, name => new_name)
    end
    return df  # Return modified DataFrame
end
replace_spaces_in_columnnames!(data)

## Transform type of columns to Float/Int
names(data, String31)
data.BEI = parse.(Float64, replace.(data.BEI, "," => ""))
data.MEI = parse.(Float64, replace.(data.MEI, "," => ""))
data.Latitude = parse.(Float64, replace.(data.Latitude, "," => ""))
data.Longitude = parse.(Float64, replace.(data.Longitude, "," => ""))
names(data, Int64)

## Normalize relevant columns (GDP)
data[:, :NormalizedGDP] = data[:, :GDP] ./ maximum(data[:, :GDP])


# Step 2: Define Agent Structure compatible with Agents.jl
mutable struct CityAgent <: AbstractAgent
    id::Int
    muni_code::String
    state::Vector{Float64}
    actions::Vector{String}
end

# Step 3: Define the model initialization function
function initialize_model(data::DataFrame, episode_length::Int)
    # Define a dummy space (non-spatial model)
    space = OpenStreetMapSpace()

    # Initialize agents
    agents = []
    for (i, row) in enumerate(eachrow(data))
        state = [
            row[:NormalizedGDP],
            row[:N._Inhabit ],
            row[:Latitude],
            row[:Longitude],
            row[:BEI_INH],
            row[:N_policies_aggregated]
        ]
        agent = CityAgent(i, string(row[:MUNI_CODE]), state, ["contribute", "defect"])
        push!(agents, agent)
    end

    # Initialize the model with agents and a defined episode length
    model = AgentBasedModel(CityAgent, space, agents; properties = Dict(:episode_length => episode_length))
    return model
end

# Step 4: Define agent behavior (step function)
function agent_step!(agent::CityAgent, model)
    action = rand(agent.actions)  # Select random action for simplicity
    if action == "contribute"
        reward = -0.05 * rand()  # Simulated reduction in CO2
    else
        reward = -0.01 * rand()  # Simulated lower reduction
    end

    println("Agent $(agent.muni_code) took action $action and received reward $reward")
    # Update agent state or model properties if needed
end

# Step 5: Run the simulation
function run_simulation!(model)
    for t in 1:model.properties[:episode_length]
        step!(model, agent_step!)  # Agent step function for each agent
    end
end

# Main execution
model = initialize_model(data, 100)
run_simulation!(model)
