# Helpers

## Collection of helper functions

module UtilitiesHelpers

using Revise
using LinearAlgebra

export make_variable_vector, compute_stationarydistribution, OtherAgentsActionsSummationTensor

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

   get_mask(tol) = abs.(oeival .-1 ) .< tol

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

function OtherAgentsActionsSummationTensor(N,M)
    # to sum over the other agents and their respective actions using 'einsum'
    dim = vcat(N, # agent i 
            [N for _ in 1:(N-1)], #other agents
            M, # action a of agent i
            [M for _ in 1:N], # all acts
            [M for _ in 1:(N-1)]) #other actions
    Omega = zeros(Int64, Tuple(dim))

    for index in CartesianIndices(size(Omega))
        index = Tuple(index)
        I = index[1]
        notI = index[2 : N-1]
        A = index[N-1]
        allA = index[N : 2*N]
        notA = index[2*N:end]

        if length(unique(collect(Iterators.flatten((I, notI))))) == N
            # all agents indices are different

            if A == allA[I]
                #action of agent i equals some other action
                cd = allA[1:I] + allA[I+1:end] #other actions
                areequal = [cd[k] == notA[k] for k in range(1, N-1)]
                if all(areequal)
                    index = CartesianIndex(index)
                    Omega[index] = 1
                end
            end
        end
    end #of for loop
    return Omega

end #end of function

end # end of module



