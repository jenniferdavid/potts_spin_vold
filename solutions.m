% nVehicles=3;
% nTasks= 6;
% 
% %input tauMat, tVec, deltaMat
% %[tauMat,tVec,deltaMat] = getTimeMat(nVehicles,nTasks);
% load("tauMat.txt");
% load("tVec.txt");
% load("deltaMat.txt");
% 
% nDim = 2*nVehicles + nTasks;
% 
% rootMat = zeros(nDim);
% maskMat = rootMat;
% for iDim = 1:nVehicles;
%     maskMat(iDim,:) = [zeros(1,nVehicles) ones(1,nTasks) zeros(1,nVehicles)];
% end
% for iDim = nVehicles+1:nDim-nVehicles
%     maskMat(iDim,:) = [zeros(1,nVehicles) ones(1,nTasks+nVehicles)];
% end
% for iDim = nDim-nVehicles+1:nDim
%     maskMat(iDim,:) = zeros(1,nDim);
% end
% for iDim = 1:nDim
%     maskMat(iDim,iDim) = 0;%diagonal elements are zero
% end
% 
% 
% subMat = zeros(nVehicles+nTasks);
% 
% 
% % Jenni code
% iMat = 0;
% v = 1:nVehicles+nTasks;
% C = perms(v);
% 
% for j=1:length(C)
%     subMat = zeros(nVehicles+nTasks);
%     
%     vMat = zeros(nDim);
%     
%     for i=1:(nVehicles+nTasks)
%         subMat(i,C(j,i))=1;
%     end
%     vMat(1:nDim-nVehicles,nVehicles+1:nDim) = subMat;
%     
%     vMat = vMat.*maskMat;
%     if(checkSums(vMat,nVehicles))
%         iMat = iMat + 1;
%         matStruct(iMat).vMat = vMat;
%     end
% end
% 
% [pMatStruct,goodPointer] = getPropagator(matStruct);
% nSolutions = sum(goodPointer); % Number of valid solutions
% nPermutations = length(goodPointer); % Number of possible matrices (including those with loops)
% 
% iGood = 0;
% k=1;
% bestCost = realmax;
% costVec = zeros(1,sum(goodPointer));
% for iMat = 1:length(goodPointer)
%     if(goodPointer(iMat))
%         iGood = iGood + 1;
%         vDeltaL = matStruct(iMat).vMat'*deltaMat;
%         vdVec = diag(vDeltaL);
%         leftVec = pMatStruct(iMat).pMat'*(tVec + vdVec);
%         vDeltaR = matStruct(iMat).vMat*deltaMat';
%         vdVec = diag(vDeltaR);
%         rightVec = pMatStruct(iMat).pMat*(tVec + vdVec);
%         
%         
%         valVec = [leftVec(nVehicles+nTasks+1:nDim,1); rightVec(1:nVehicles)];
%         costVec(1,iGood) = max(valVec);
%         if(costVec(1,iGood) < bestCost)
%             bestCost = costVec(1,iGood);
%             vMatBest = matStruct(iMat).vMat;
%            % matStruct2(k).vMatBest = vMatBest;
%             k=k+1;
%         end
%     end
% end
% 
% %sol = deltaMat.*matStruct2(2).vMatBest;
% %sum(sum(sol))
% 
% %exTime = toc;            
% fprintf('----------------------------\n');
% fprintf('%d vehicles with %d tasks\n',[nVehicles nTasks]);
% fprintf('Lowest cost = %.2f\n',bestCost);
% fprintf('Maximum cost =%.2f, average = %.2f, stddev = %.2f\n\n',[max(costVec) mean(costVec) std(costVec)]);
% fprintf('Best V matrix:\n');
% [nR,nC] = size(vMatBest);
% for iR = 1:nR
%     for iC = 1:nC
%         fprintf('%d\t',vMatBest(iR,iC));
%     end
%     fprintf('\n');
% end
% 
% fprintf('\n');
% [solStrA,solStrB,checkTime] = getSolutionStrings(vMatBest,tVec,deltaMat,nVehicles,nTasks);
% fprintf('This corresponds to the following routing:\n')
% fprintf('%s\n',solStrA);
% fprintf('%s = ',solStrB);
% fprintf('%.2f\n',checkTime);
% fprintf('\n');
% fprintf('Task times:\n');
% for iR = 1:nR
%      fprintf('%.2f\t',tVec(iR));
% end
% fprintf('\n');
% fprintf('Delta matrix:\n');
% for iR = 1:nR
%     for iC = 1:nC
%          fprintf('%.2f\t',deltaMat(iR,iC));
%     end
%     fprintf('\n');
% end
% fprintf('\n');
% fprintf('Complexity:\n')
% fprintf('%d possible matrices, %d valid matrices\n',[nPermutations nSolutions])
% fprintf('\n');
% VV = load("VMatrix.txt");
% x = VV(:,1);
% y = VV(:,2:end);
% plot(x,y);
%             
%           %  pause
% 
% % 
% v = 21
% 
% VV = load("VMatrix");
% x = VV(1:end,1);
% y = VV(1:end,2:end);
% figure(1);
% plot(x,y);
% title('VMatrix')
% grid;
% 
% 
% % VV = load("VMatrix");
% % x = VV(1:end,1);
% % y = VV(3000:4000,v);
% % figure(2);
% % plot(x,y);
% % title('VMatrix-11')
% % grid;
% 
% VV = load("VMatrix");
% x = VV(1:end,1);
% y = VV(1:end,v);
% figure(3);
% plot(x,y);
% title('VMatrix-11')
% grid;
% 
% 
% VV = load("VMatrix");
% x = VV(5100:5300,1);
% y = VV(5100:5300,v);
% figure(4);
% plot(x,y);
% title('VMatrix')
% grid;
% 
% 
% RR = load("Etask.txt");
% x = RR(1:end,1);
% y = RR(1:end,2:end);
% figure(5);
% plot(x,y);
% title('Etask')
% grid;
% 
% 
% LL = load("leftVec.txt");
% x = LL(1:end,1);
% y = LL(1:end,2:end);
% figure(6);
% plot(x,y);
% title('leftVec')
% grid;


H = load("hist");
iterations = H(1:end,1); 
max = H(1:end, 2);
X = [max,iterations];
hist3(X);
xlabel('iteration'); ylabel('max');
set(gcf,'renderer','opengl');
%%
% Color the bars based on the frequency of the observations, i.e. according
% to the height of the bars.
set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
