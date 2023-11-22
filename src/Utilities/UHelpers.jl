# Helpers

## Collection of helper functions

module UtilitiesHelpers

using Revise
using LinearAlgebra

export make_variable_vector, compute_stationarydistribution

function make_variable_vector(variable, len::Int)
    #"""Turn a `variable` into a vector or check that `length` is consistent."""  
    if variable isa Number 
        return repeat([variable], len) # Repeat to create a vector of specified length
    else
        @assert length(variable) == len "Wrong number given"
        return collect(variable)  # Convert to vector
    end
end #of function

function compute_stationarydistribution(Tkk::AbstractArray)
   #"""Compute stationary distribution for transition matrix `Tkk`."""
   # eigenvectors
   oeival = real(eigvals(transpose(Tkk)))
   oeivec = real(eigvecs(transpose(Tkk)))

   get_mask = tol -> abs.(oeival .-1 ) .< tol

   tolerances = map(x -> 0.1^x, 1:15)
   masks = map(get_mask, tolerances)
   ixrange = range(1, length(masks))[sum.(masks).≥ 1]
   mask = if isempty(ixrange) masks[end] else masks[findmax(ixrange)[1]] end
   tol = if isempty(ixrange) tolerances[end] else tolerances[findmax(ixrange)[1]] end

   # obtain stationary distribution
   meivec = transpose(ifelse.(mask, oeivec, -42))

   dist = meivec ./ sum(meivec, dims=1)

   dist = ifelse.(dist .< tol, 0, dist)
   dist = dist ./ sum(dist, dims=1)

   return ifelse.(meivec .== -42, -10, dist)
   
end #of function








end # end of module



