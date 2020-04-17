
% Maximum likelihood training of a 2-layer MLP
% assuming additive (white) Gaussian noise
close all, 
clear
dummyOut = 0;
% Input N specifies number of training samples

F = 10;
Perceptrons = 10;
numberOfClasses = 1;
numberOfSamples = 3;
numberOfDtrain_1000 = 1000; 
numberOfDtest = 10000;


%Generate Dtrain and Dtest
Dtrain_1000 = exam4q1_generateData(numberOfDtrain_1000);
Dtest_10000 = exam4q1_generateData(numberOfDtest);

fig = 0;
fig = fig + 1;
figure(fig), plot(Dtrain_1000(1,:),Dtrain_1000(2,:),'.'),
xlabel('X_1'); ylabel('X_2');


fig = fig + 1;
figure(fig), plot(Dtest_10000(1,:),Dtest_10000(2,:),'.'),
xlabel('X_1'); ylabel('X_2');
%Use a for loop to iterate three datasets

numberOfDtrain = numberOfDtrain_1000;
Dtrain = Dtrain_1000(1,:);
Dtest = Dtest_10000(1,:);

Ylabels= Dtrain_1000(2,:);
YtestLabels = Dtest_10000(2,:);

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

MSE = zeros(F, Perceptrons);
MSEAverage = zeros(1, Perceptrons);
%Initialize a struct to store params result at the last 10fold operations
ParamsStore = struct();
FoldParamsStore = struct();
counter = 1;
 for nPerceptrons = 1:Perceptrons
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

        type = "SOFTPLUS";

    
        yValidate = Ylabels(validateIndex);
        xValidate = Dtrain(validateIndex);
        validateLen = length(validateIndex);

        yTrain = Ylabels(trainIndex);
        xTrain = Dtrain(trainIndex);
        trainLen = length(trainIndex);
    
        X = xTrain;
        Y = yTrain;  
        nX = size(X,1); 
        nY = size(Y,1);
        
        params.A = randn(nPerceptrons,nX);
        params.b = randn(nPerceptrons,1);
        params.C = randn(nY,nPerceptrons);
        params.d = mean(Y,2);
        
        %disp(["xTrain", size(xTrain)])
        %Determine/specify sizes of parameter matrices/vectors

        sizeParams = [nX;nPerceptrons;nY];
        
        %Initialize model parameters
        %zeros(nY,1); % initialize to mean of y

        %params = paramsTrue;

        vecParamsInit = [params.A(:);params.b;params.C(:);params.d];

        %Optimize model
        options = optimset('MaxFunEvals',10000, 'MaxIter',10000); %Increase MaxFunEvals and MaxIter
        vecParams = fminsearch(@(vecParams)(objectiveFunction(type, X,Y,sizeParams,vecParams)),vecParamsInit, options);

        
        params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
        params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
        params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
        params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
    
        H = mlpModel(type, xValidate,params);
        
        N = size(xValidate,2);
        MSE(k, nPerceptrons) = sum(sum((yValidate-H).*(yValidate-H),1),2)/N;
        
        %Calculate error percentage
        %[val, testIdx] = max(H);
        %[val, labelIdx] = max(yValidate);
        %error = find(testIdx~=labelIdx);
%         disp("H");
%         disp(H(1:10));
%         disp("yValidate");
%         disp(yValidate(1:10));
%         errorP = size(error)/size(xValidate);
%         errorTrain(k, nPerceptrons) = errorP;
        
        FoldParamsStore(k, nPerceptrons).A = params.A;
        FoldParamsStore(k, nPerceptrons).b = params.b;
        FoldParamsStore(k, nPerceptrons).C = params.C;
        FoldParamsStore(k, nPerceptrons).d = params.d;
        
    end
    disp([counter,"/11"])
    MSEAverage(1, nPerceptrons) = mean(MSE(:, nPerceptrons))
    %Calculate error Average
%     errorAverage(af, nPerceptrons) = mean(errorTrain(:, nPerceptrons));
%     errorTrain(:, nPerceptrons);
%     [val, minK] = min(errorTrain(:, nPerceptrons));
%     minK = min(minK(:))
    %Store best parms for fulture usage
%     ParamsStore(af,nPerceptrons).A = FoldParamsStore(minK, nPerceptrons).A;
%     ParamsStore(af,nPerceptrons).b = FoldParamsStore(minK, nPerceptrons).b;
%     ParamsStore(af,nPerceptrons).C = FoldParamsStore(minK, nPerceptrons).C;
%     ParamsStore(af,nPerceptrons).d = FoldParamsStore(minK, nPerceptrons).d;
    clear FoldParamsStore
 end

 
 MSEAverage
 
%  for nPerceptrons = 1:Perceptrons
%     for af = 1:numberOfaf
%          bar(1:Perceptrons, errorAverage(af, nPerceptrons))
%     end
%  end
 fig = fig + 1;
 figure(fig), clf,
 b1 = bar(1:nPerceptrons, MSEAverage),
 title("MSE for model trained with each perceptrons"),
 ylabel('MSE'), 
 xlabel('Numbers of perceptrons'), 
 legend('SOFTPLUS'), 
 drawnow()
 
 [val, idx] = min(MSEAverage(:))
 nPerceptrons =find(MSEAverage==val)
 
 
 nPerceptrons = max(nPerceptrons(:))% In case same perceptrons performance


type = "SOFTPLUS";

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
options = optimset('MaxFunEvals',10000, 'MaxIter',10000);
vecParams = fminsearch(@(vecParams)(objectiveFunction(type, X,Y,sizeParams,vecParams)),vecParamsInit, options);


%Visualize model output for training data
params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
H = mlpModel(type, Dtest,params);
%Calculate error percentage
%[val, testIdx] = max(H);
%[val, labelIdx] = max(yValidate);
%error = find(testIdx~=labelIdx);

N = size(Dtest,2);
MSEFinal = sum(sum((YtestLabels-H).*(YtestLabels-H),1),2)/N
size(H)
fig = fig + 1;
figure(fig), plot(Dtest_10000(1,:),Dtest_10000(2,:),'.'),hold on
plot(Dtest_10000(1,:),H,'x')
xlabel('X_1'); ylabel('X_2');
legend('Original Data', 'Estimated Data'), 
drawnow()
% error = find(H~=YtestLabels);
% errorP = size(error)/size(xValidate);
% disp(["Final Accuracy:", 1-errorP])


  
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
H = V; % linear output layer activations
%H = exp(V)./repmat(sum(exp(V),1),nY,1); % softmax nonlinearity for second/last layer
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