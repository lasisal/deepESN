function internalWeights = generate_internal_SCR_weights(nInternalUnits)
% GENERATE_INTERNAL_SCR_WEIGHTS creates a SCR reservoir for an ESN
%  
% inputs:
% nInternalUnits = the number of internal units in the ESN
% 
% output:
% internalWeights = matrix of size nInternalUnits x nInternalUnits
% internalWeights(i,j-1) = r -> 1
% internalWeights(1,nInternalUnits) = r 

%
% Created May 24, 2018, X. Liu

r = 0.99;
internalWeights = zeros(nInternalUnits,nInternalUnits);
for i=2:nInternalUnits
    internalWeights(i,i-1) = r;
end;
internalWeights(1,nInternalUnits) = r;

