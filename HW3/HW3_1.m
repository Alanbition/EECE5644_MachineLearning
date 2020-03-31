
% Maximum likelihood training of a 2-layer MLP
% assuming additive (white) Gaussian noise
close all, 
clear
dummyOut = 0;
% Input N specifies number of training samples

F = 10;
Perceptrons = 10;
numberOfClasses = 3;
numberOfSamples = 3;
numberOfDtrain_100 = 100; 
numberOfDtrain_500 = 500; 
numberOfDtrain_1000 = 1000; 
numberOfDtest = 10000;


%Generate Dtrain and Dtest
[Dtrain_100,DtrainLabels_100] = generateMultiringDataset(numberOfClasses,numberOfDtrain_100);
[Dtrain_500,DtrainLabels_500] = generateMultiringDataset(numberOfClasses,numberOfDtrain_500);
[Dtrain_1000,DtrainLabels_1000] = generateMultiringDataset(numberOfClasses,numberOfDtrain_1000);
[Dtest,DtestLabels] = generateMultiringDataset(numberOfClasses,numberOfDtest);
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

%Use a for loop to iterate three datasets
for s=1:numberOfSamples
    disp(strcat('Question1 with sample',num2str(s)));
    if s == 1
        numberOfDtrain = numberOfDtrain_100;
        Dtrain = Dtrain_100;
        DtrainLabels = DtrainLabels_100;
    elseif s == 2
        numberOfDtrain = numberOfDtrain_500;
        Dtrain = Dtrain_500;
        DtrainLabels = DtrainLabels_500;      
    else
        numberOfDtrain = numberOfDtrain_1000;
        Dtrain = Dtrain_1000;
        DtrainLabels = DtrainLabels_1000;
    end
    
    colors = rand(numberOfClasses,3);
    fig = fig+1;
    figure(fig), clf,
    for l = 1:numberOfClasses
        ind_l = find(DtrainLabels==l);
        plot(Dtrain(1,ind_l),Dtrain(2,ind_l),'.','MarkerFaceColor',colors(l,:)), axis equal, hold on,
    end
    xlabel('x1'); ylabel('x2');
    title(strcat('Sample data from Generated Dtrain size = ', num2str(numberOfDtrain)));
    drawnow()

%Seperate Dtrain and Dtest data to a new array with the format:
%(Class number, 1 or 0)

for c=1:numberOfClasses
    Ylabels(c,:)=DtrainLabels==c;
    YtestLabels(c,:)=DtestLabels==c;
end

%Split the data set to 10 block for 10 fold
block = ceil(linspace(0, numberOfDtrain, F+1));

for k = 1:F
    datasetBlock(k,:) = [block(k)+1,block(k+1)];
end
% Intialize likelihood
% 
% 
% ----------------------------------Start of Question 1------------------------------------
numberOfaf = 3;

%Initialize array to store errors
errorTrain = zeros(F, Perceptrons);
errorValidate = zeros(F, Perceptrons);
errorAverage = zeros(3,Perceptrons);

%Initialize a struct to store params result at the last 10fold operations
ParamsStore = struct();
FoldParamsStore = struct();
counter = 1;
 for nPerceptrons = 1:Perceptrons
    for af = 1:numberOfaf
        %Initializae params here so that next fold can use the previous
        %result(which result a better accuracy than put inside 10fold loop from my testing)

        counter = counter + 1;
        for k = 1:F
            
        %Assign validation and train set for each iteration
        validateIndex = [datasetBlock(k,1):datasetBlock(k,2)];
        if k == 1
            trainIndex = [datasetBlock(k, 2)+1:numberOfDtrain];
        elseif k == F
            trainIndex = [1:datasetBlock(k, 1)-1];
        else
            trainIndex = [1:datasetBlock(k-1, 2), datasetBlock(k+1, 2):numberOfDtrain];
        end
    
        %Select active function from sigmod, ISRU and SOFTPLUS
        if af == 1
            type = "sigmod";
        elseif af == 2
            type = "ISRU";
        else
            type = "SOFTPLUS";
        end
    
        yValidate = [Ylabels(1,validateIndex);Ylabels(2,validateIndex);Ylabels(3,validateIndex)];
        xValidate = [Dtrain(1,validateIndex);Dtrain(2, validateIndex)];
        validateLen = length(validateIndex);

        yTrain = [Ylabels(1,trainIndex);Ylabels(2,trainIndex);Ylabels(3,trainIndex)];
        xTrain = [Dtrain(1,trainIndex);Dtrain(2, trainIndex)];
        trainLen = length(trainIndex);
    
        X = xTrain;
        Y = yTrain;  

        params.A = randn(nPerceptrons,2);
        params.b = randn(nPerceptrons,1);
        params.C = randn(3,nPerceptrons);
        params.d = mean(Y,2);
        
        %disp(["xTrain", size(xTrain)])
        %Determine/specify sizes of parameter matrices/vectors
        nX = size(X,1); 
        nY = size(Y,1);
        sizeParams = [nX;nPerceptrons;nY];
        
        %Initialize model parameters
        %zeros(nY,1); % initialize to mean of y

        %params = paramsTrue;

        vecParamsInit = [params.A(:);params.b;params.C(:);params.d];

        %Optimize model
        options = optimset('MaxFunEvals',200000, 'MaxIter',200000); %Increase MaxFunEvals and MaxIter
        vecParams = fminsearch(@(vecParams)(objectiveFunction(type, X,Y,sizeParams,vecParams)),vecParamsInit, options);

        
        params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
        params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
        params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
        params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
    
        H = mlpModel(type, xValidate,params);
        
        %Calculate error percentage
        [val, testIdx] = max(H);
        [val, labelIdx] = max(yValidate);
        error = find(testIdx~=labelIdx);
        errorP = size(error)/size(testIdx);
        errorTrain(k, nPerceptrons) = errorP;
        FoldParamsStore(k, nPerceptrons).A = params.A;
        FoldParamsStore(k, nPerceptrons).b = params.b;
        FoldParamsStore(k, nPerceptrons).C = params.C;
        FoldParamsStore(k, nPerceptrons).d = params.d;
        
    end
    disp([counter,"/31"])
    %Calculate error Average
    errorAverage(af, nPerceptrons) = mean(errorTrain(:, nPerceptrons));
    errorTrain(:, nPerceptrons);
    [val, minK] = min(errorTrain(:, nPerceptrons));
    minK = min(minK(:))
    %Store best parms for fulture usage
    ParamsStore(af,nPerceptrons).A = FoldParamsStore(minK, nPerceptrons).A;
    ParamsStore(af,nPerceptrons).b = FoldParamsStore(minK, nPerceptrons).b;
    ParamsStore(af,nPerceptrons).C = FoldParamsStore(minK, nPerceptrons).C;
    ParamsStore(af,nPerceptrons).d = FoldParamsStore(minK, nPerceptrons).d;
    clear FoldParamsStore
    end
 end
 
 errorAverage;
 
%  for nPerceptrons = 1:Perceptrons
%     for af = 1:numberOfaf
%          bar(1:Perceptrons, errorAverage(af, nPerceptrons))
%     end
%  end
 fig = fig + 1;
 figure(fig), clf,
 b1 = bar(1:nPerceptrons, errorAverage),
 title(strcat('Probability of error for model trained with each activation functions and perceptrons with Dtrain size = ', num2str(numberOfDtrain))),
 ylabel('Probability of error'), 
 xlabel('Numbers of perceptrons'), 
 legend('Sigmod','ISRU', 'SOFTPLUS'), 
 drawnow()
 
 [val, idx] = min(errorAverage(:));
 [af, nPerceptrons] =find(errorAverage==val);
 
 
 af = min(af(:));% In case select same af performance
 nPerceptrons = max(nPerceptrons(:));% In case same perceptrons performance

 BestMLP(s,1) = af;
 BestMLP(s,2) = nPerceptrons;
 
if af == 1
    type = "sigmod";
elseif af == 2
    type = "ISRU";
else
    type = "SOFTPLUS";
end

X = Dtrain;%Y
Y = Ylabels;%training label
%Determine/specify sizes of parameter matrices/vectors
nX = size(X,1); 
nY = size(Y,1);
sizeParams = [nX;nPerceptrons;nY];
%Initialize model parameters

params.A = randn(nPerceptrons,nX);
params.b = randn(nPerceptrons,1);
params.C = randn(nY,nPerceptrons);
params.d = mean(Y,2);

%Init with pervious best params
% params.A = ParamsStore(af,nPerceptrons).A;
% params.b = ParamsStore(af,nPerceptrons).b;
% params.C = ParamsStore(af,nPerceptrons).C;
% params.d = mean(Y,2);%ParamsStore(af,nPerceptrons).d;


vecParamsInit = [params.A(:);params.b;params.C(:);params.d];

%Optimize mode
options = optimset('MaxFunEvals',200000, 'MaxIter',200000);
vecParams = fminsearch(@(vecParams)(objectiveFunction(type, X,Y,sizeParams,vecParams)),vecParamsInit, options);


%Visualize model output for training data
params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
H = mlpModel(type, Dtest,params);
[val, testIdx] = max(H);
[val, labelIdx] = max(YtestLabels);
error = find(testIdx~=labelIdx);
errorP = size(error)/size(testIdx);
errorStore(s) = 1- size(error)/size(testIdx);
disp(["Final Accuracy:", 1-errorP])
clear Ylabels
clear YtestLabels
clear block
clear datasetBlock

end

 fig = fig +1;
 figure(fig), clf,
 X = categorical({'100','500','1000'});
 X = reordercats(X,{'100','500','1000'});
 b1 = bar(X, errorStore),
 title('Validation testing dataset on MLP model trained with 100, 500, 1000 samples of data'),
 ylabel('Percentage of correctly classified instances'), 
 xlabel('Numbers of samples'), 
 drawnow()

% %----------------------------------End of Question 1------------------------------------


%----------------------------------Start of Question 2------------------------------------


delta = 1e-5; % tolerance for EM stopping criterion
regWeight = 1e-11; % regularization parameter for covariance estimates
%Get seperate data from 3 classes

%We have 3 class, the the number of row of mu should be 2
mu_true = [-8 -8 8;-8 8 8];%This is just a dummy for row = 2
[d,M] = size(mu_true);




for s=1:numberOfSamples
    
    disp(strcat('Question2 with sample',num2str(s)));
    if s == 1
        numberOfDtrain = numberOfDtrain_100;
        Dtrain = Dtrain_100;
        DtrainLabels = DtrainLabels_100;
    elseif s == 2
        numberOfDtrain = numberOfDtrain_500;
        Dtrain = Dtrain_500;
        DtrainLabels = DtrainLabels_500;      
    else
        numberOfDtrain = numberOfDtrain_1000;
        Dtrain = Dtrain_1000;
        DtrainLabels = DtrainLabels_1000;
    end
    
    for c=1:numberOfClasses
        Ylabels(c,:)=DtrainLabels==c;
        YtestLabels(c,:)=DtestLabels==c;
    end

    %Split the data set to 10 block for 10 fold
    block = ceil(linspace(0, numberOfDtrain, F+1));

    for k = 1:F
        datasetBlock(k,:) = [block(k)+1,block(k+1)];
    end


    %Splite the data to 3 class, along with the 10 blocks for each class
    for c=1:numberOfClasses
        if c== 1
            index1 = find(Ylabels(c,:));
            class1Train = Dtrain(:,index1);
            [row, col] = size(class1Train);
            block1 = ceil(linspace(0, col, F+1));
            for k = 1:F
                datasetBlock1(k,:) = [block1(k)+1,block1(k+1)];
            end
        elseif c == 2
            index2 = find(Ylabels(c,:));
            class2Train = Dtrain(:,index2);
            [row, col] = size(class2Train);
            block2 = ceil(linspace(0, col, F+1));
            for k = 1:F
                datasetBlock2(k,:) = [block2(k)+1,block2(k+1)];
            end
        else
            index3 = find(Ylabels(c,:));
            class3Train = Dtrain(:,index3);
            [row, col] = size(class3Train);
            block3 = ceil(linspace(0, col, F+1));
            for k = 1:F
                datasetBlock3(k,:) = [block3(k)+1,block3(k+1)];
            end
        end
    end


    %Intialize likelihood for eval
    likelihoodTrain = zeros(F, 6);
    likelihoodValidate = zeros(F, 6);
    Averagelltrain = zeros(numberOfClasses,6); 
    Averagellvalidate = zeros(numberOfClasses,6);

    %1-6 Gaussian components
    for c = 1:numberOfClasses
        if c == 1
            datasetBlock = datasetBlock1;
            x = class1Train;
        elseif c == 2
            datasetBlock = datasetBlock2;
            x = class2Train;        
        else
            datasetBlock = datasetBlock3;
            x = class3Train;
        end
        
    
        for M = 1:6 
            for k = 1:F

            [row, col] = size(x);
            N = col;
            % Assign validation and train set for each iteration
            validateIndex = [datasetBlock(k,1):datasetBlock(k,2)];

            if k == 1
                trainIndex = [datasetBlock(k, 2)+1:N];
            elseif k == F
                trainIndex = [1:datasetBlock(k, 1)-1];
            else
                trainIndex = [1:datasetBlock(k-1, 2), datasetBlock(k+1, 2):N];
            end

            xValidate = [x(1,validateIndex);x(2, validateIndex)];
            validateLen = length(validateIndex);

            xTrain = [x(1,trainIndex);x(2, trainIndex)];
            trainLen = length(trainIndex);

    
        % Initialize the GMM to randomly selected samples
        alpha = ones(1,M)/M;
        shuffledIndices = randperm(trainLen);
        mu = xTrain(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
        [~,assignedCentroidLabels] = min(pdist2(mu',xTrain'),[],1); % assign each sample to the nearest mean
        for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
            Sigma(:,:,m) = cov(xTrain(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
        end
        t = 0; %displayProgress(t,x,alpha,mu,Sigma);
    
        Converged = 0; % Not converged at the beginning
        for i = 1:5000 %At least 100
            for l = 1:M
                temp(l,:) = repmat(alpha(l),1,trainLen).*evalGaussian(xTrain,mu(:,l),Sigma(:,:,l));
            end
            plgivenx = temp./sum(temp,1);
            clear temp
            alphaNew = mean(plgivenx,2);
            w = plgivenx./repmat(sum(plgivenx,2),1,trainLen);
            muNew = xTrain*w';
            for l = 1:M
                v = xTrain-repmat(muNew(:,l),1,trainLen);
                u = repmat(w(l,:),d,1).*v;
                SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
            end
            Dalpha = sum(abs(alphaNew-alpha'));
            Dmu = sum(sum(abs(muNew-mu)));
            DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
            Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
            if Converged
                break
            end
            alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
            t = t+1;
            %displayProgress(t,xTrain,alpha,mu,Sigma);
        end
        likelihoodTrain(k,M) = sum(log(evalGMM(xTrain,alpha,mu,Sigma)));
        likelihoodValidate(k,M) = sum(log(evalGMM(xValidate,alpha,mu,Sigma)));    
        end
        %Store the average likelihood in an matrix
        AverageTrain(c,M) = mean(likelihoodTrain(:,M)); 
        AverageValidate(c,M) = mean(likelihoodValidate(:,M));
        %Clear the values after each 10 fold
        clear Sigma
        clear mu
        clear alpha
        clear SigmaNew
        clear muNew
        clear alphaNew
    % If there is any inf number in likelihood, replace it with the
    % minimum likelihood in validation set
        if isinf(AverageValidate(c,M))
            AverageValidate(c,M) = (min(AverageValidate(c,(find(isfinite(AverageValidate(c,:)))))));
        end
    end
end

%Disp result
AverageValidate;


 fig = fig + 1;
 figure(fig), clf,
 b1 = bar(1:6, AverageValidate),
 title(strcat('Validation Log-Likelihoods for Dtrain size = ',num2str(numberOfDtrain))),
 xlabel('GMM Order'),
 ylabel(strcat('Log likelihood estimate')),
 legend('class1','class2', 'class3'), 
 drawnow()
 
[val, testIdx] = max(AverageValidate,[], 2);

%Disp the best GMM model order for each class

BestGMM(s,:) = testIdx;

%create a struct to store the mean, covariance and prior
GMMStore = struct();


%Train with the whole Dtrain dataset and store the component for each classes
for c = 1:numberOfClasses
    if c == 1
        x = class1Train;
    elseif c == 2
        x = class2Train;        
    else
        x = class3Train;
    end
    M = testIdx(c, 1);
    [row, col] = size(x);
    N = col;

    xTrain = x;
    trainLen = length(x);


% Initialize the GMM to randomly selected samples
    alpha = ones(1,M)/M;
    shuffledIndices = randperm(trainLen);
    mu = xTrain(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
    [~,assignedCentroidLabels] = min(pdist2(mu',xTrain'),[],1); % assign each sample to the nearest mean
    for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
        Sigma(:,:,m) = cov(xTrain(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
    end
    t = 0; %displayProgress(t,x,alpha,mu,Sigma);

    Converged = 0; % Not converged at the beginning
        for i = 1:10000 %At least 100
            for l = 1:M
                temp(l,:) = repmat(alpha(l),1,trainLen).*evalGaussian(xTrain,mu(:,l),Sigma(:,:,l));
            end
            plgivenx = temp./sum(temp,1);
            clear temp
            alphaNew = mean(plgivenx,2);
            w = plgivenx./repmat(sum(plgivenx,2),1,trainLen);
            muNew = xTrain*w';
            for l = 1:M
                v = xTrain-repmat(muNew(:,l),1,trainLen);
                u = repmat(w(l,:),d,1).*v;
                SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
            end
            Dalpha = sum(abs(alphaNew-alpha'));
            Dmu = sum(sum(abs(muNew-mu)));
            DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
            Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
            if Converged
                break
            end
            alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
            t = t+1;
            %displayProgress(t,xTrain,alpha,mu,Sigma);
        end
%likelihoodTrain(k,M) = sum(log(evalGMM(xTrain,alpha,mu,Sigma)));
%likelihoodValidate(k,M) = sum(log(evalGMM(xValidate,alpha,mu,Sigma)));

        %Store GMM component here in GMMStore struct
        GMMStore(c).alpha = alpha;
        GMMStore(c).mu = mu;
        GMMStore(c).Sigma = Sigma;
        clear Sigma
        clear mu
        clear alpha
        clear SigmaNew
        clear muNew
        clear alphaNew
    end






    %Seperate the Dtest data and calculate the priors for each class
    for c=1:numberOfClasses
        if c== 1
            index1 = find(YtestLabels(c,:));
            class1Test = Dtest(:,index1);
            [row1, col1] = size(class1Test);
            p(c) = col1/numberOfDtest;
        elseif c == 2
            index2 = find(YtestLabels(c,:));
            class2Test = Dtest(:,index2);
            [row2, col2] = size(class2Test);
            p(c) = col2/numberOfDtest;
        else
            index3 = find(YtestLabels(c,:));
            class3Test = Dtest(:,index3);
            [row3, col3] = size(class3Test);
            p(c) = col3/numberOfDtest;
        end
    end


    %Store the GMM eval result for each class into GMMProbability array
    for i = 1:3
        alpha = GMMStore(i).alpha;
        mu = GMMStore(i).mu;
        Sigma = GMMStore(i).Sigma;
        GMMProbability(i, :) = evalGMM(Dtest,alpha,mu,Sigma).*p(i); %P(X|Thelta) * P(Thelta)
        Prior(s,i) = p(i);
    end

%Apply MAP to find which class has the highest probability result and
%compare with the test labels
[val, testIdx] = max(GMMProbability);
[val, labelIdx] = max(YtestLabels);
error = find(testIdx~=labelIdx);
errorP = size(error)/size(testIdx);
AccuracyP(s) = 1-size(error)/size(testIdx);
disp(["Final Accuracy GMM:", 1-errorP])
clear Ylabels
clear YtestLabels
clear block
clear datasetBlock
clear Sigma
clear mu
clear alpha
clear SigmaNew
clear muNew
clear alphaNew
end

 fig = fig +1;
 figure(fig), clf,
 X = categorical({'100','500','1000'});
 X = reordercats(X,{'100','500','1000'});
 b1 = bar(X, AccuracyP),
 title('Validation testing dataset on GMM model trained with 100, 500, 1000 samples of data'),
 ylabel('Percentage of correctly classified instances'), 
 xlabel('Numbers of samples'), 
 drawnow()

 
 
  BestMLP
  
  BestGMM
  
  Prior

  errorStore
  
  AccuracyP
  
% BestMLP =
% 
%      1    10
%      3    10
%      3     8
% 
% 
% BestGMM =
% 
%      3     1     2
%      3     3     6
%      3     4     6
% 
% 
% Prior =
% 
%     0.3327    0.3297    0.3376
%     0.3327    0.3297    0.3376
%     0.3327    0.3297    0.3376
% 
% 
% errorStore =
% 
%     0.5091    0.9602    0.9613
% 
% 
% AccuracyP =
% 
%     0.9208    0.9670    0.9717


  
function objFncValue = objectiveFunction(type, X,Y,sizeParams,vecParams)
N = size(X,2); % number of samples
nX = sizeParams(1);
nPerceptrons = sizeParams(2);
nY = sizeParams(3);
params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
H = mlpModel(type, X,params);
objFncValue = sum(sum((Y-H).*(Y-H),1),2)/N;
%objFncValue = sum(-sum(Y.*log(H),1),2)/N;
% Change objective function to make this MLE for class posterior modeling
end

%
function H = mlpModel(type, X,params)
N = size(X,2);                          % number of samples
nY = length(params.d);                  % number of outputs
U = params.A*X + repmat(params.b,1,N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
Z = activationFunction(type, U);              % z \in R^nP, using nP instead of nPerceptons
V = params.C*Z + repmat(params.d,1,N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
%H = V; % linear output layer activations
H = exp(V)./repmat(sum(exp(V),1),nY,1); % softmax nonlinearity for second/last layer
% Add softmax layer to make this a model for class posteriors
%
end

function out = activationFunction(type, in)
if type == "sigmod"
    out = 1./(1+exp(-in)); % logistic function
elseif type == "ISRU"
    out = in./sqrt(1+in.^2); % ISRU
else
    out = log(1+exp(in));% Soft Plus
end
end

function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end

%%%
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end
%%%
function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
zGMM = reshape(GMM,91,101);
%figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
end
%%%
function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end
%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end