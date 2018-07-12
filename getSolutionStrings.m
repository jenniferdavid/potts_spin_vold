function [solStrA,solStrB,checkTime] = getSolutionStrings(vMatBest,tVec,deltaMat,nVehicles,nTasks)

% [solStrA,solStrB,checkTime] = getSolutionStrings(vMatBest,tVec,deltaMat,nVehicles,nTasks);

vecLetters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
taskLetters = 'abcdefghijklmnopqrstuvwxyz';

nDim = size(vMatBest,1);
cTime = zeros(1,nVehicles);
for iV = 1:nVehicles
    indx = find(vMatBest(iV,:) == 1);
    if(iV == 1)
        solStrA = ['S' vecLetters(iV) ' -> ' taskLetters(indx-nVehicles)];
        solStrB = ['max(' num2str(deltaMat(iV,indx)) ' + ' num2str(tVec(indx))];
    else
        solStrA = [solStrA ' & S' vecLetters(iV) ' -> ' taskLetters(indx-nVehicles)];
        solStrB = [solStrB ', ' num2str(deltaMat(iV,indx)) ' + ' num2str(tVec(indx))];
    end
    cTime(iV) = deltaMat(iV,indx) + tVec(indx);
        
    while(indx <= nDim-nVehicles)
        indxB = find(vMatBest(indx,:) == 1);
        if(indxB > nVehicles+nTasks)
            solStrA = [solStrA ' -> E' vecLetters(indxB-nVehicles-nTasks)];
            solStrB = [solStrB ' + ' num2str(deltaMat(indx,indxB))];
            cTime(iV) = cTime(iV) + deltaMat(indx,indxB);
            solStrB = [solStrB ' = ' num2str(cTime(iV))];
        else
            solStrA = [solStrA ' -> ' taskLetters(indxB-nVehicles)];
            solStrB = [solStrB ' + ' num2str(deltaMat(indx,indxB)) ' + ' num2str(tVec(indxB))];
            cTime(iV) = cTime(iV) + deltaMat(indx,indxB) + tVec(indxB);
        end
        indx = indxB;
    end
end
solStrB = [solStrB ')'];

checkTime = max(cTime);

