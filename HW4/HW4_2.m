% assuming additive (white) Gaussian noise
close all, 
clear
dummyOut = 0;
% Input N specifies number of training samples

F = 10;
Perceptrons = 10;
numberOfClasses = 2;
numberOfDtrain_1000 = 1000; 
numberOfDtest = 10000;


%Generate Dtrain and Dtest
[Dtrain_1000,DtrainLabels_1000] = generateMultiringDataset(numberOfClasses,numberOfDtrain_1000);
[Dtest,DtestLabels] = generateMultiringDataset(numberOfClasses,numberOfDtest);

%Plot Dtest
fig = 1
figure(fig), clf,
colors = rand(numberOfDtest,3);
for l = 1:numberOfClasses
    ind_l = find(DtestLabels==l);
    plot(Dtest(1,ind_l),Dtest(2,ind_l),'.','MarkerFaceColor',colors(l,:)), axis equal, hold on,
end
xlabel('x1'); ylabel('x2');
title(strcat('Sample data from Generated Dtest = ', num2str(numberOfDtest)));
drawnow()

%Plot Dtrain
fig = fig + 1;
figure(fig), clf,
colors = rand(numberOfDtest,3);
for l = 1:numberOfClasses
    ind_l = find(DtrainLabels_1000==l);
    plot(Dtrain_1000(1,ind_l),Dtrain_1000(2,ind_l),'.','MarkerFaceColor',colors(l,:)), axis equal, hold on,
end
xlabel('x1'); ylabel('x2');
title(strcat('Sample data from Generated Dtrain = ', num2str(numberOfDtrain_1000)));
drawnow()



for c=1:numberOfClasses
        index1 = find(DtestLabels==c);
        pTest(c) = size(index1,2)/numberOfDtest;
end

for c=1:numberOfClasses
        index2 = find(DtrainLabels_1000==c);
        pTrain(c) = size(index2,2)/numberOfDtrain_1000;
end

disp(pTest)
disp(pTrain)

DtrainLabels_1000 = DtrainLabels_1000-1;
l = 2*(DtrainLabels_1000-0.5);
x = Dtrain_1000;
N=1000; n = 2; K=10;

DtestLabels = DtestLabels-1;
lTest = 2*(DtestLabels-0.5);
xTest = Dtest;
% N=1000; n = 2; K=10;
% mu(:,1) = [-1;0]; mu(:,2) = [1;0]; 
% Sigma(:,:,1) = [2 0;0 1]; Sigma(:,:,2) = [1 0;0 4];
% p = [0.35,0.65]; % class priors for labels 0 and 1 respectively
% % Generate samples
% label = rand(1,N) >= p(1); l = 2*(label-0.5);
% Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
% x = zeros(n,N); % reserve space
% % Draw samples from each class pdf
% for lbl = 0:1
%     x(:,label==lbl) = randGaussian(Nc(lbl+1),mu(:,lbl+1),Sigma(:,:,lbl+1));
% end


% Train a Gaussian kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)

dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
CList = 10.^linspace(-1,9,11)
sigmaList = 10.^linspace(-2,3,13)

for sigmaCounter = 1:length(sigmaList)
    [sigmaCounter,length(sigmaList)],
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xValidate = x(:,indValidate); % Using folk k as validation set
            lValidate = l(indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
            end
            % using all other folds as training set
            xTrain = x(:,indTrain); lTrain = l(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','RBF','KernelScale',sigma);
            dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
            indCORRECT = find(lValidate.*dValidate == 1); 
            Ncorrect(k)=length(indCORRECT);
        end 
        PCorrect(CCounter,sigmaCounter)= sum(Ncorrect)/N;
    end 
end

PCorrect(CCounter,sigmaCounter)

 fig = fig + 1;
 figure(fig), clf,
 b1 = bar(1:sigmaCounter, PCorrect),
 title('Probability of correct classification for model trained with each C and Sigma'),
 ylabel('Probability of correct classification'), 
 xlabel('Sigma Value'), 
  for iN = 1:length(CList)
      legendCell{iN} = num2str(CList(iN),'C = %e');
  end
  legend(legendCell), 
 drawnow()

 
 fig = fig + 1;
 figure(fig), clf,
 b1 = bar(1:CCounter, PCorrect'),
 title('Probability of correct classification for model trained with each Sigma and C'),
 ylabel('Probability of correct classification'), 
 xlabel('C Value'), 
  for iN = 1:length(sigmaList)
      legendCell2{iN} = num2str(sigmaList(iN),'Sigma = %e');
  end
  legend(legendCell2), 
 drawnow()


fig = fig + 1;
figure(fig), subplot(1,2,1),
contour(log10(CList),log10(sigmaList),PCorrect',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest= CList(indBestC) 
sigmaBest= sigmaList(indBestSigma)
SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','RBF','KernelScale',sigmaBest);



d = SVMBest.predict(xTest')'; % Labels of training data using the trained SVM
indINCORRECT = find(lTest.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(lTest.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(fig), subplot(1,2,2), 
plot(xTest(1,indCORRECT),xTest(2,indCORRECT),'g.'), hold on,
plot(xTest(1,indINCORRECT),xTest(2,indINCORRECT),'r.'), axis equal,
title('Testing Data (RED: Incorrectly Classified)'),
pTrainingError = length(indINCORRECT)/numberOfDtest, % Empirical estimate of training error probability
Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(fig), subplot(1,2,2), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,