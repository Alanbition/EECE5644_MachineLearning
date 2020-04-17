clear all, close all,


filenames{1,2} = '3096_color.jpg';
filenames{1,1} = '42049_color.jpg';

Kvalues = [2,3,4]; % desired numbers of clusters



for imageCounter = 1:2 %size(filenames,2)
    imdata = imread(filenames{1,imageCounter}); 
    figure(1), subplot(size(filenames,2),length(Kvalues)+1,(imageCounter-1)*(length(Kvalues)+1)+1), imshow(imdata);
    [R,C,D] = size(imdata); N = R*C; imdata = double(imdata);
    rowIndices = [1:R]'*ones(1,C); colIndices = ones(R,1)*[1:C];
    features = [rowIndices(:)';colIndices(:)']; % initialize with row and column indices
    for d = 1:D
        imdatad = imdata(:,:,d); % pick one color at a time
        features = [features;imdatad(:)'];
    end
    minf = min(features,[],2); maxf = max(features,[],2);
    ranges = maxf-minf;
    x = diag(ranges.^(-1))*(features-repmat(minf,1,N)); % each feature normalized to the unit interval [0,1]
    disp(size(x))




        %(1) 2 components
        x = x';
        options = statset('MaxIter',3000, 'TolFun',1e-5);
        GMM = fitgmdist(x, 2,'Options',options,'RegularizationValue',1e-11);
        disp(size(GMM));
        p = posterior(GMM, x);
        disp(size(p));
        [~, FirstImgIndx] = max(p, [], 2);
        disp(size(FirstImgIndx));
        figure
        lbls = reshape(FirstImgIndx,R, C);
        imagesc(lbls);
        colormap(hsv(2));
        colorbar('Ticks',1:2);
        
        clear p
        clear GMM
        clear lbls
        
        
        x = x';
        %(2) 2-6 components
        %Split the data set to 10 block for 10 fold
        F = 10;
        block = ceil(linspace(0, size(x,2), F+1));

        for k = 1:F
            datasetBlock(k,:) = [block(k)+1,block(k+1)];
        end        
        
        likelihoodValidate = zeros(F, 3);
        AverageValidate = zeros(1,3);       

        for p = 1:10
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

                xValidate = [x(1,validateIndex);x(2, validateIndex);x(3, validateIndex);x(4, validateIndex);x(5, validateIndex)];
                validateLen = length(validateIndex);

                xTrain = [x(1,trainIndex);x(2, trainIndex);x(3, trainIndex);x(4, trainIndex);x(5, trainIndex)];
                
                trainLen = length(trainIndex);  
               for M = 2:4
                    options = statset('MaxIter',3000, 'TolFun',1e-5);
                    GMM = fitgmdist(xTrain', M,'Options',options,'RegularizationValue',1e-11);
                    alpha = GMM.ComponentProportion;
                    mu = (GMM.mu)';
                    sigma = GMM.Sigma;
                    likelihoodValidate(F,M-1) = sum(log(evalGMM(xValidate, alpha, mu, sigma)));
               end
        end
        end
        
        AverageValidate = sum(likelihoodValidate)/F
 

         figure, clf,
         b1 = bar(2:4, AverageValidate),
         title('Validation Log-Likelihoods'),
         xlabel('Cluster'),
         ylabel(strcat('Log likelihood estimate')),
         drawnow()
 
        [val, BestIdx] = max(AverageValidate,[], 2)
        BestGMM = BestIdx+1
        
        x = x';
        options = statset('MaxIter',3000, 'TolFun',1e-5);
        GMM = fitgmdist(x, BestGMM, 'Options',options,'RegularizationValue',1e-11);
        p = posterior(GMM, x);
        [~, newImgIndex] = max(p, [], 2);
        
        figure
        lbls = reshape(newImgIndex,R, C);
        imagesc(lbls);
        colormap(hsv(BestGMM));
        colorbar('Ticks',1:BestGMM);
        
        clear likelihoodValidate
        clear GMM
        clear xTrain
        clear xValidate
        clear datasetBlock
        clear AverageValidate
        clear newImgIndex
        clear x
end




% 
% Apply MAP to find which class has the highest probability result and
% compare with the test labels
% [val, testIdx] = max(GMMProbability);
% [val, labelIdx] = max(YtestLabels);
% error = find(testIdx~=labelIdx);
% errorP = size(error)/size(testIdx);
% AccuracyP(s) = 1-size(error)/size(testIdx);
% 
% 
%     
    
 

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
    
    
    
    