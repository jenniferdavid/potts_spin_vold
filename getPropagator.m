function [pMatStruct,goodPointer] = getPropagator(matStruct)

% function [pMatStruct,goodPointer] = getPropagator(matStruct)
%
% Returns the propagators and tells how many that are infinite

nMat = size(matStruct,2);
nDim = size(matStruct(1).vMat,1);

goodPointer = zeros(1,nMat);

isBad =0;
for iMat = 1:nMat
    Pinv = eye(nDim) - matStruct(iMat).vMat;
    P = inv(Pinv);
    if(any(isinf(P(1,:))))
        pMatStruct(iMat).pMat = Inf;
        fprintf('vMat %d is singular\n',iMat);
        isBad = isBad + 1;
    else
        pMatStruct(iMat).pMat = P;
        goodPointer(iMat) = 1;
    end
end

goodPointer = logical(goodPointer);

fprintf('There were %d (out of %d) illegal solutions\n',[isBad nMat]);
