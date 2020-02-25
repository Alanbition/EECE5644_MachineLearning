%% Source code from  Aravind H. M. ("Arvin") in volunteer g drive folder

clear; close all; clc;
%% Input and initializations
N = 10;                       % Number of samples
mu = [1;0];                     % Mean - sample
Sigma = [1,0.3;0.3,1];          % Covariance - sample
SigmaV = 2*0.5;                 % V8ariance - 0-mean Gaussian noise
nRealizations = 100;            % Number of realizations for the ensemble analysis

gammaArray = 10.^[-10:0.1:5];   % Array of gamma values

A = [0.4 -0.3;-0.3 0.8];        % Coefficients for quadratic terms
b = [-0.14; 0.97];              % Coefficients for linear terms 
c = -2;                         % Constant

% True parameter array
params = [c;b(1);b(2);A(1,1);A(1,2)+A(2,1);A(2,2)];

%% MAP parameter estimation for an ensemble set of samples
tic;
clearvars -except params mu Sigma SigmaV gammaArray nRealizations;
[msqError, avMsqError, avPercentError, avAbsPercentError] = deal(zeros(nRealizations,length(gammaArray)));
for n = 1:nRealizations
    N = 10;

    x = (-1 + 2.*rand(N,2))';

    % Calculate y: quadratic in x + additive 0-mean Gaussian noise
    yTruth{1,n} = yFunc(x,params);
    y = yTruth{1,n} + SigmaV^0.5*randn(1,N);
    zQ = [ones(1,size(x,2)); x(1,:); x(2,:); x(1,:).^2; x(1,:).*x(2,:); x(2,:).^2];

    % Compute z*z^T for linear and quadratic models
    for i = 1:N; zzTQ(:,:,i) = zQ(:,i)*zQ(:,i)'; end
    
    % MAP parameter estimation
    for i = 1:length(gammaArray)
        gamma = gammaArray(i);
        thetaMAP{1,n}(:,i) = (sum(zzTQ,3)+SigmaV/gamma^2*eye(size(zQ,1)))^-1*sum(repmat(y,size(zQ,1),1).*zQ,2);
        yMAP{1,n}(:,i) = yFunc(x,thetaMAP{1,n}(:,i));
    end
    
    % Mean squared error in y
    msqError(n,:) = N\sum((yMAP{1,n}-repmat(yTruth{1,n}',1,length(gammaArray))).^2,1);
    
    % Average mean squared error of estimated parameters
    avMsqError(n,1:length(gammaArray)) = length(params)\sum((thetaMAP{1,n} - ...
        repmat(params,1,length(gammaArray))).^2);%./repmat(params,1,length(gammaArray))*100,1);
    
end
toc;

%% Plot results - MAP Ensemble: mean squared error
fig = figure; fig.Position([1,2]) = [50,100];
fig.Position([3 4]) = 1.5*fig.Position([3,4]);
percentileArray = [5,25,50,75,95];

ax = gca; hold on; box on;
prctlMsqError = prctile(avMsqError,percentileArray,1);
p=plot(ax,gammaArray,prctlMsqError,'LineWidth',2);
xlabel('gamma'); ylabel('average mean squared error of parameters'); ax.XScale = 'log';
lgnd = legend(ax,p,[num2str(percentileArray'),...
    repmat(' percentile',length(percentileArray),1)]); lgnd.Location = 'southwest';
%pause;

[~,ind] = min(abs(prctlMsqError(3,:)));
plot(ax,gammaArray(ind),prctlMsqError(3,ind),'ro');
lgnd = legend(ax,p,[num2str(percentileArray'),...
    repmat(' percentile',length(percentileArray),1)]); lgnd.Location = 'southwest';
%pause;
    

%% Function to calculate y (without noise), given x and parameters
function y = yFunc(x,params)
    A = [params(4),params(5)/2;params(5)/2,params(6)];
    b = [params(2);params(3)];
    c = params(1);
    y = diag(x'*A*x)' + b'*x + c;
end