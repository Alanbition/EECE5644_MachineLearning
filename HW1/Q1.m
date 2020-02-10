%For question 1 - 1
disp("Question 1 Problem 1")
% Expected risk minimization with 2 classes
clear all, close all,

n = 2; % number of feature dimensions
N = 10000; % number of iid sampless
mu(:,1) = [-0.1;0]; mu(:,2) = [0.1;0];
Sigma(:,:,1) = [1 -0.9;-0.9 1]; Sigma(:,:,2) = [1 0.9;0.9 1];
p = [0.8,0.2]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);% if p larger or equal to 0.8=>0, else 1


Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space two dim size N Matrix
% Draw samples from each class pdf
for l = 0:1
    %x(:,label==l) = randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1));
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
figure(1), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 

discriminantScores = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));

thresholds = sort(discriminantScores);
thresholdFinal = nan(1, length(thresholds)+1);
thresholdFinal(1) = thresholds(1) - 0.05;
thresholdFinal(end) = thresholds(end) + 0.05;
for k = 1:length(thresholdFinal)-2
    thresholdFinal(k+1)  = (thresholds(k)+thresholds(k+1))/2;
end

tp = nan(1, length(thresholdFinal));
fp = nan(1, length(thresholdFinal));
perror = nan(1, length(thresholdFinal));

for i = 1:numel(thresholdFinal)
    %Change threshold from 0 to infinity
    discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
    decision = (discriminantScore >= log(thresholdFinal(end-i+1)));
    ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
    ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive  
    tp(i) = p11;
    fp(i) = p10;
    perror(i) = [p10,p01]*Nc'/N;
end
[M, I] = min(perror);
disp(["Min Perror is " M]);
disp(["Best threshold is " thresholdFinal(end-I+1)]);
disp(["tp is " tp(I)]);
disp(["fp is " fp(I)]);
figure(2)
plot(fp, tp);
hold on
plot(fp(I), tp(I), 'r*');

title('ROC curve'),
xlabel('False Positive Rate'), ylabel('True Positive Rate'), 




%For question 1 - problem 3
disp("Question 1 Problem 1_FisherLDA")

% Appending LDA to the ERM code for TakeHomeQ3...
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly
figure(3), clf,
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
legend('Class 0','Class 1'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 
tau = 0;
%decisionLDA = (yLDA >= 0);
yLDAs = sort(yLDA);
yLDAFinal = nan(1, length(yLDAs)+1);
yLDAFinal(1) = yLDAs(1) - 0.05;
yLDAFinal(end) = yLDAs(end) + 0.05;
for k = 1:length(yLDAFinal)-2
    yLDAFinal(k+1)  = (yLDAs(k)+yLDAs(k+1))/2;
end

tp = nan(1, length(yLDAFinal));
fp = nan(1, length(yLDAFinal));
perror = nan(1, length(yLDAFinal));

for i = 1:numel(yLDAFinal)
    %Change threshold from 0 to infinity
    %discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
    decision = (yLDA >= yLDAFinal(end-i+1));
    ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
    ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive  
    tp(i) = p11;
    fp(i) = p10;
    perror(i) = [p10,p01]*Nc'/N;;
end
[M, I] = min(perror);
disp(["Min Perror is " M]);
disp(["Best threshold is " yLDAFinal(end-I+1)]);
disp(["tp is " tp(I)]);
disp(["fp is " fp(I)]);

figure(4)
plot(fp, tp);
hold on
plot(fp(I), tp(I), 'r*');

title('Problem1 FisherLDA ROC curve'),
xlabel('False Positive Rate'), ylabel('True Positive Rate'), 

%For question 1 - problem 2
disp("Question 1 Problem 2")
% Expected risk minimization with 2 classes

%n = 2; % number of feature dimensions
%N = 1000; % number of iid samples
%mu(:,1) = [-0.1;0]; mu(:,2) = [0.1;0];
Sigma2(:,:,1) = eye(2); Sigma2(:,:,2) = eye(2);
%p = [0.8,0.2]; % class priors for labels 0 and 1 respectively
%label = rand(1,N) >= p(1);% if p larger or equal to 0.8=>0, else 1


%Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
%disp(["Nc is " Nc]);
x2 = zeros(n,N); % save up space two dim size N Matrix
% Draw samples from each class pdf
for l = 0:1
    %x(:,label==l) = randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1));
    x2(:,label==l) = mvnrnd(mu(:,l+1),Sigma2(:,:,l+1),Nc(l+1))';
end
figure(5), clf,
plot(x2(1,label==0),x2(2,label==0),'o'), hold on,
plot(x2(1,label==1),x2(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 

discriminantScores2 = log(evalGaussian(x2,mu(:,2),Sigma2(:,:,2)))-log(evalGaussian(x2,mu(:,1),Sigma2(:,:,1)));

thresholds2 = sort(discriminantScores2);
thresholdFinal2 = nan(1, length(thresholds2)+1);
thresholdFinal2(1) = thresholds2(1) - 0.05;
thresholdFinal2(end) = thresholds2(end) + 0.05;
for k = 1:length(thresholdFinal2)-2
    thresholdFinal2(k+1)  = (thresholds2(k)+thresholds2(k+1))/2;
end

tp2 = nan(1, length(thresholdFinal2));
fp2 = nan(1, length(thresholdFinal2));
perror2 = nan(1, length(thresholdFinal2));

for i = 1:numel(thresholdFinal2)
    %Change threshold from 0 to infinity
    discriminantScore2 = log(evalGaussian(x2,mu(:,2),Sigma2(:,:,2)))-log(evalGaussian(x2,mu(:,1),Sigma2(:,:,1)));% - log(gamma);
    decision2 = (discriminantScore2 >= log(thresholdFinal2(end-i+1)));
    ind00_2 = find(decision2==0 & label==0); p00_2 = length(ind00_2)/Nc(1); % probability of true negative
    ind10_2 = find(decision2==1 & label==0); p10_2 = length(ind10_2)/Nc(1); % probability of false positive
    ind01_2 = find(decision2==0 & label==1); p01_2 = length(ind01_2)/Nc(2); % probability of false negative
    ind11_2 = find(decision2==1 & label==1); p11_2 = length(ind11_2)/Nc(2); % probability of true positive  
    tp2(i) = p11_2;
    fp2(i) = p10_2;
    perror2(i) = [p10_2,p01_2]*Nc'/N;;
end
[M2, I2] = min(perror2);
disp(["Min Perror is " M2]);
disp(["Best threshold is " thresholdFinal2(end-I2+1)]);
disp(["tp is " tp2(I2)]);
disp(["fp is " fp2(I2)]);
figure(6)
plot(fp2, tp2);
hold on
plot(fp2(I2), tp2(I2), 'r*');

title('ROC curve for Naive-Bayesian Classifier'),
xlabel('False Positive Rate'), ylabel('True Positive Rate'), 



%For question 1 - problem 3
disp("Question 1 Problem 2_FisherLDA")

% Appending LDA to the ERM code for TakeHomeQ3...
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma2(:,:,1) + Sigma2(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly
figure(7), clf,
plot(yLDA(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
plot(yLDA(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
legend('Class 0','Class 1'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 
tau = 0;
%decisionLDA = (yLDA >= 0);
yLDAs = sort(yLDA);
yLDAFinal = nan(1, length(yLDAs)+1);
yLDAFinal(1) = yLDAs(1) - 0.05;
yLDAFinal(end) = yLDAs(end) + 0.05;
for k = 1:length(yLDAFinal)-2
    yLDAFinal(k+1)  = (yLDAs(k)+yLDAs(k+1))/2;
end

tp = nan(1, length(yLDAFinal));
fp = nan(1, length(yLDAFinal));
perror = nan(1, length(yLDAFinal));

for i = 1:numel(yLDAFinal)
    %Change threshold from 0 to infinity
    %discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
    decision = (yLDA >= yLDAFinal(end-i+1));
    ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
    ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive  
    tp(i) = p11;
    fp(i) = p10;
    perror(i) = [p10,p01]*Nc'/N;;
end
[M, I] = min(perror);
disp(["Min Perror is " M]);
disp(["Best threshold is " yLDAFinal(end-I+1)]);
disp(["tp is " tp(I)]);
disp(["fp is " fp(I)]);
figure(8)
plot(fp, tp);
hold on
plot(fp(I), tp(I), 'r*');

title('Problem2 FisherLDA ROC curve'),
xlabel('False Positive Rate'), ylabel('True Positive Rate'), 


