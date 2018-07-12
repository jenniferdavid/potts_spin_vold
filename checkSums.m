function isNorm = checkSums(vMat,nVehicles)

% function isNorm = checkSums(vMat,nVehicles)
%
% Checks the relevant row and column sums

[nRows,nCols] = size(vMat);
if(nRows ~= nCols)
    error('Matrix must be square')
end

colSums = sum(vMat,1);
rowSums = sum(vMat,2);

isNorm = logical(1);
if(any(colSums(nVehicles+1:nCols) ~= 1))
    isNorm = logical(0);
end
if(any(rowSums(1:nRows-nVehicles) ~= 1))
    isNorm = logical(0);
end

