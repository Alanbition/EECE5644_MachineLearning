
% Generates N samples from a specified GMM,
% then uses EM algorithm to estimate the parameters
% of a GMM that has the same nu,mber of components
% as the true GMM that generates the samples.

N= 1000;% Number of data
close all,
delta = 1e-5; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
F = 10;


% Generate samples from a 4-component GMM
alpha_true = [0.2,0.25,0.3,0.35];
mu_true = [-8 -8 8 8;-8 8 8 -8];
Sigma_true(:,:,1) = [24 1;1 18];
Sigma_true(:,:,2) = [25 3;3 9];
Sigma_true(:,:,3) = [14 -5;-5 16];
Sigma_true(:,:,4) = [6 1;1 20];
x = randGMM(N,alpha_true,mu_true,Sigma_true);


[d,M] = size(mu_true); % determine dimensionality of samples and number of GMM components

%plot the sample data
figure(1),
plot(x(1,:), x(2,:),'ro')
xlabel('x1'); ylabel('x2');
title(strcat('Sample data from GMM N = ', num2str(N)));

%Split the data set to 10 block
block = ceil(linspace(0, N, F+1));
for k = 1:F
    datasetBlock(k,:) = [block(k)+1,block(k+1)];
end

%Intialize likelihood
likelihoodTrain = zeros(F, 6);
likelihoodValidate = zeros(F, 6);
Averagelltrain = zeros(1,6); Averagellvalidate = zeros(1,6);

%6 Gaussian components
for MM = 1:100
for M = 1:6 
    for k = 1:F
    
        
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
    for i = 1:100 %At least 100
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
        AverageTrain(1,M) = mean(likelihoodTrain(:,M)); 
        AverageValidate(1,M) = mean(likelihoodValidate(:,M));
        
        % If there is any inf number in likelihood, replace it with the
        % minimum likelihood in validation set
        if isinf(AverageValidate(1,M))
            AverageValidate(1,M) = (min(AverageValidate(find(isfinite(AverageValidate)))));
        end   
end
end
AverageTrain
AverageValidate


figure(2), clf,
plot(AverageTrain,'bo', 'MarkerSize', 15); 
xlabel('GMM Order'); ylabel(strcat('Log likelihood estimate'));
title(strcat('Training Log-Likelihoods for N=',num2str(N)));
grid on

figure(3), clf,
plot(AverageValidate,'r*','MarkerSize', 15);
xlabel('GMM Order'); ylabel(strcat('Log likelihood estimate'));
title(strcat('Validation Log-Likelihoods for N=',num2str(N)));
grid on
%%%
function displayProgress(t,x,alpha,mu,Sigma)
figure(1),
if size(x,1)==2
    subplot(1,2,1), cla,
    plot(x(1,:),x(2,:),'b.'); 
    xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
    axis equal, hold on;
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
    contour(x1Grid,x2Grid,zGMM); axis equal, 
    subplot(1,2,2), 
end
logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));
plot(t,logLikelihood,'b.'); hold on,
xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
drawnow; pause(0.1),
end

%%%
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