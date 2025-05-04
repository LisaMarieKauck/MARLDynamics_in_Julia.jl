using XLSX
using Agents
using ReinforcementLearning

# Define the CityAgent struct
mutable struct CityAgent <: AbstractAgent
    id::Int
    country::String
    population::Int
    latitude::Float64
    longitude::Float64
    climate::Int
    GDP::Int
    BEI::Float64
    BEI_Inh::Int
    MEI::Float64
    MEI_Inh::Int
    N_policies::Int
    N_education::Int
    N_self_gov::Int
    N_finance::Int
    N_regulation::Int
    N_other::Int
end

# Function to load city agents from an Excel file
function load_city_agents(file_path::String, sheet_name::String="2databrief-gis-monicities")
    data = XLSX.readxlsx(file_path)
    data_table = data[sheet_name]
    agents = []

    for row in XLSX.eachtablerow(data_table)
        #row = XLSX.row_number(r)
        id = row[1]
        country = row[2]
        population = row[3]
        latitude = row[:Latitude]
        longitude = row[:Longitude]
        climate = row[6]
        GDP = row[:GDP]
        BEI = row[:BEI]
        BEI_Inh = row[:BEI_INH]
        MEI = row[:MEI]
        MEI_Inh = row[:MEI_INH]
        N_policies = row[13]
        N_education = row[14]
        N_self_gov = row[15]
        N_finance = row[16]
        N_regulation = row[17]
        N_other = row[18]
        push!(agents, CityAgent(id, country, population, latitude, longitude, climate, GDP, BEI, BEI_Inh, MEI, MEI_Inh, N_policies, N_education, N_self_gov, N_finance, N_regulation, N_other))
    end

    return agents
end

# Load city agents from an Excel file
file_path = "src/Agents/Data/databrief-signatories.xlsx"
city_agents = load_city_agents(file_path, "2databrief-gis-monicities")

# Print loaded agents to verify
for agent in city_agents
    println("ID: $(agent.id), Country: $(agent.country), Population: $(agent.population), Latitude: $(agent.latitude), Longitude: $(agent.longitude)")
end

# defining a grid

size = (20, 20)
space = GridSpaceSingle(size; periodic = false, metric = :chebyshev)

# Define the CityAgent struct
@agent struct CityAgent_2(GridAgent{2})
    number::Int
    country::String
    population::Int
    latitude::Float64
    longitude::Float64
    climate::Int
    GDP::Int
    BEI::Float64
    BEI_Inh::Int
    MEI::Float64
    MEI_Inh::Int
    N_policies::Int
    N_education::Int
    N_self_gov::Int
    N_finance::Int
    N_regulation::Int
    N_other::Int
end

# Function to load city agents from an Excel file
function load_city_agents(file_path::String, sheet_name::String="2databrief-gis-monicities")
    data = XLSX.readxlsx(file_path)
    data_table = data[sheet_name]
    agents = []

    for row in XLSX.eachtablerow(data_table)
        #row = XLSX.row_number(r)
        number = row[1]
        country = row[2]
        population = row[3]
        latitude = row[:Latitude]
        longitude = row[:Longitude]
        climate = row[6]
        GDP = row[:GDP]
        BEI = row[:BEI]
        BEI_Inh = row[:BEI_INH]
        MEI = row[:MEI]
        MEI_Inh = row[:MEI_INH]
        N_policies = row[13]
        N_education = row[14]
        N_self_gov = row[15]
        N_finance = row[16]
        N_regulation = row[17]
        N_other = row[18]
        push!(agents, CityAgent_2(number, country, population, latitude, longitude, climate, GDP, BEI, BEI_Inh, MEI, MEI_Inh, N_policies, N_education, N_self_gov, N_finance, N_regulation, N_other))
    end

    return agents
end

# Load city agents from an Excel file
file_path = "src/Agents/Data/databrief-signatories.xlsx"
city_agents = load_city_agents(file_path, "2databrief-gis-monicities")

# Print loaded agents to verify
for agent in city_agents
    println("ID: $(agent.number), Country: $(agent.country), Population: $(agent.population), Latitude: $(agent.latitude), Longitude: $(agent.longitude)")
end
