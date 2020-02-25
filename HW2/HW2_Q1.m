%% Source code from Anja Deric from volunteer g drive folder
%For question 1 Part 1
disp("Question 1 Part 1")
% Expected risk minimization with 2 classes
clear all, close all,

%% ====================== Generate Test Data Set ====================== %%

N_test = 10000;
n = 2; % number of feature dimensions
mu(:,2) = [-2;0]; mu(:,1) = [2;0];
Sigma(:,:,2) = [1 -0.9;-0.9 2]; Sigma(:,:,1) = [2 0.9;0.9 1];
p = [0.9,0.1]; % class priors for labels 0 and 1 respectively

% Generating true class labels
label_test = (rand(1,N_test) >= p(1))';
Nc_test = [length(find(label_test==0)),length(find(label_test==1))];

% Draw samples from each class pdf
x_test = zeros(N_test,n); 

for L = 0:1
    x_test(label_test==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc_test(L+1));
end
disp(size(x_test));

x_testp1 = x_test';
label_testp1 = label_test';


figure(1), clf,
plot(x_testp1(1, label_testp1==0),x_testp1(2, label_testp1==0),'o'), hold on,
plot(x_testp1(1, label_testp1==1),x_testp1(2, label_testp1==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 

discriminantScores = log(evalGaussian(x_testp1,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x_testp1,mu(:,1),Sigma(:,:,1)));
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
    decision = (discriminantScores >= log(thresholdFinal(end-i+1)));
    ind00 = find(decision==0 & label_testp1==0); p00 = length(ind00)/Nc_test(1); % probability of true negative
    ind10 = find(decision==1 & label_testp1==0); p10 = length(ind10)/Nc_test(1); % probability of false positive
    ind01 = find(decision==0 & label_testp1==1); p01 = length(ind01)/Nc_test(2); % probability of false negative
    ind11 = find(decision==1 & label_testp1==1); p11 = length(ind11)/Nc_test(2); % probability of true positive  
    tp(i) = p11;
    fp(i) = p10;
    perror(i) = [p10,p01]*Nc_test'/N_test;
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

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScores = log(evalGaussian(x_testp1,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x_testp1,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScores >= log(gamma));

ind00 = find(decision==0 & label_testp1==0); p00 = length(ind00)/Nc_test(1); % probability of true negative
ind10 = find(decision==1 & label_testp1==0); p10 = length(ind10)/Nc_test(1); % probability of false positive
ind01 = find(decision==0 & label_testp1==1); p01 = length(ind01)/Nc_test(2); % probability of false negative
ind11 = find(decision==1 & label_testp1==1); p11 = length(ind11)/Nc_test(2); % probability of true positive
%p(error) = [p10,p01]*Nc_test'/N; % probability of error, empirically estimated

perror = [p10,p01]*Nc_test'/N_test;
disp(["perror" perror]);

% plot correct and incorrect decisions
figure(3), % class 0 circle, class 1 +, correct green, incorrect red
plot(x_testp1(1,ind00),x_testp1(2,ind00),'og'); hold on,
plot(x_testp1(1,ind10),x_testp1(2,ind10),'or'); hold on,
plot(x_testp1(1,ind01),x_testp1(2,ind01),'+r'); hold on,
plot(x_testp1(1,ind11),x_testp1(2,ind11),'+g'); hold on,
axis equal,

% Draw the decision boundary
horizontalGrid = linspace(floor(min(x_testp1(1,:))),ceil(max(x_testp1(1,:))),101);
verticalGrid = linspace(floor(min(x_testp1(2,:))),ceil(max(x_testp1(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
figure(3), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ,'Location','southeast'), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 


%% ================== Generate and Plot Training Set N=10 ================== %%

N = 10;   % number of iid samples

% Generating true class labels
label = (rand(1,N) >= p(1))';
Nc = [length(find(label==0)),length(find(label==1))];

% Draw samples from each class pdf
x = zeros(N,n); 
for L = 0:1
    x(label==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc(L+1));
end

%Plot samples with true class labels
figure(4);
plot(x(label==0,1),x(label==0,2),'o',x(label==1,1),x(label==1,2),'+');
legend('Class 0','Class 1'); title('Training Data and True Class Labels');
xlabel('x_1'); ylabel('x_2'); hold on;

%% ======================== Logistic Regression  N=10======================= %%
% Initialize fitting parameters
x = [ones(N, 1) x];
initial_theta = zeros(n+1, 1);
label=double(label);

% Compute gradient descent to get theta values
[theta, cost] = gradient_descent(x,N,label,initial_theta,1,1000);
[theta2, cost2] = fminsearch(@(t)(cost_func(t, x, label, N)), initial_theta);

% Choose points to draw boundary line
plot_x1 = [min(x(:,2))-2,  max(x(:,2))+2];                      
plot_x2(1,:) = (-1./theta(3)).*(theta(2).*plot_x1 + theta(1));  
plot_x2(2,:) = (-1./theta2(3)).*(theta2(2).*plot_x1 + theta2(1)); % fminsearch

% Plot decision boundary
plot(plot_x1, plot_x2(1,:), plot_x1, plot_x2(2,:));  
axis([plot_x1(1), plot_x1(2), min(x(:,3))-2, max(x(:,3))+2]);
legend('Class 0', 'Class 1', ' Classifier (from scratch)', 'Classifier (fminsearch)');

% Plot cost function
figure(5); plot(cost);
title('Calculated Cost');
xlabel('Iteration number'); ylabel('Cost');


%% ========================= Test Classifier  N=10 ========================== %%
% Coefficients for decision boundary line equation
coeff(1,:) = polyfit([plot_x1(1), plot_x1(2)], [plot_x2(1,1), plot_x2(1,2)], 1);
coeff(2,:) = polyfit([plot_x1(1), plot_x1(2)], [plot_x2(2,1), plot_x2(2,2)], 1); %fminsearch
% Decide based on which side of the line each point is on
for i = 1:2
    if coeff(i,1) >= 0
        decision_p2(:,i) = (coeff(i,1).*x_test(:,1) + coeff(i,2)) < x_test(:,2);
    else
        decision_p2(:,i) = (coeff(i,1).*x_test(:,1) + coeff(i,2)) > x_test(:,2);
    end
end

% error1 = plot_test_data(decision_p2(:,1), label_test, Nc_test, p, 6, x_test, plot_x1, plot_x2(1,:));
% title('Test Data Classification (from scratch)');
% fprintf('Total error (classifier from scratch): %.2f%%\n',error1);

error2 = plot_test_data(decision_p2(:,2), label_test, Nc_test, p, 7, x_test, plot_x1, plot_x2(2,:));
title('Test Data Classification (using fminsearch)');
fprintf('Total error (classifier using fminsearch): %.2f%%\n',error2);


%% ================== Generate and Plot Training Set N=100 ================== %%

N = 100;   % number of iid samples

% Generating true class labels
label = (rand(1,N) >= p(1))';
Nc = [length(find(label==0)),length(find(label==1))];

% Draw samples from each class pdf
x = zeros(N,n); 
for L = 0:1
    x(label==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc(L+1));
end

%Plot samples with true class labels
figure(8);
plot(x(label==0,1),x(label==0,2),'o',x(label==1,1),x(label==1,2),'+');
legend('Class 0','Class 1'); title('Training Data and True Class Labels');
xlabel('x_1'); ylabel('x_2'); hold on;

%% ======================== Logistic Regression  N=100======================= %%
% Initialize fitting parameters
x = [ones(N, 1) x];
initial_theta = zeros(n+1, 1);
label=double(label);


% Compute gradient descent to get theta values
[theta, cost] = gradient_descent(x,N,label,initial_theta,1,1000);
[theta2, cost2] = fminsearch(@(t)(cost_func(t, x, label, N)), initial_theta);

% Choose points to draw boundary line
plot_x1 = [min(x(:,2))-2,  max(x(:,2))+2];                      
plot_x2(1,:) = (-1./theta(3)).*(theta(2).*plot_x1 + theta(1));  
plot_x2(2,:) = (-1./theta2(3)).*(theta2(2).*plot_x1 + theta2(1)); % fminsearch

% Plot decision boundary
plot(plot_x1, plot_x2(1,:), plot_x1, plot_x2(2,:));  
axis([plot_x1(1), plot_x1(2), min(x(:,3))-2, max(x(:,3))+2]);
legend('Class 0', 'Class 1', ' Classifier (from scratch)', 'Classifier (fminsearch)');

% Plot cost function
figure(9); plot(cost);
title('Calculated Cost');
xlabel('Iteration number'); ylabel('Cost');


%% ========================= Test Classifier  N=100 ========================== %%
% Coefficients for decision boundary line equation
coeff(1,:) = polyfit([plot_x1(1), plot_x1(2)], [plot_x2(1,1), plot_x2(1,2)], 1);
coeff(2,:) = polyfit([plot_x1(1), plot_x1(2)], [plot_x2(2,1), plot_x2(2,2)], 1); %fminsearch
% Decide based on which side of the line each point is on
for i = 1:2
    if coeff(i,1) >= 0
        decision_p2(:,i) = (coeff(i,1).*x_test(:,1) + coeff(i,2)) < x_test(:,2);
    else
        decision_p2(:,i) = (coeff(i,1).*x_test(:,1) + coeff(i,2)) > x_test(:,2);
    end
end

% error1 = plot_test_data(decision_p2(:,1), label_test, Nc_test, p, 10, x_test, plot_x1, plot_x2(1,:));
% title('Test Data Classification (from scratch)');
% fprintf('Total error (classifier from scratch): %.2f%%\n',error1);

error2 = plot_test_data(decision_p2(:,2), label_test, Nc_test, p, 11, x_test, plot_x1, plot_x2(2,:));
title('Test Data Classification (using fminsearch)');
fprintf('Total error (classifier using fminsearch): %.2f%%\n',error2);


%% ================== Generate and Plot Training Set N=1000 ================== %%

N = 1000;   % number of iid samples

% Generating true class labels
label = (rand(1,N) >= p(1))';
Nc = [length(find(label==0)),length(find(label==1))];

% Draw samples from each class pdf
x = zeros(N,n); 
for L = 0:1
    x(label==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc(L+1));
end

%Plot samples with true class labels
figure(12);
plot(x(label==0,1),x(label==0,2),'o',x(label==1,1),x(label==1,2),'+');
legend('Class 0','Class 1'); title('Training Data and True Class Labels');
xlabel('x_1'); ylabel('x_2'); hold on;

%% ======================== Logistic Regression  N=1000======================= %%
% Initialize fitting parameters
x = [ones(N, 1) x];
initial_theta = zeros(n+1, 1);
label=double(label);

% Compute gradient descent to get theta values
[theta, cost] = gradient_descent(x,N,label,initial_theta,1,1000);
[theta2, cost2] = fminsearch(@(t)(cost_func(t, x, label, N)), initial_theta);

% Choose points to draw boundary line
plot_x1 = [min(x(:,2))-2,  max(x(:,2))+2];                      
plot_x2(1,:) = (-1./theta(3)).*(theta(2).*plot_x1 + theta(1));  
plot_x2(2,:) = (-1./theta2(3)).*(theta2(2).*plot_x1 + theta2(1)); % fminsearch

% Plot decision boundary
plot(plot_x1, plot_x2(1,:), plot_x1, plot_x2(2,:));  
axis([plot_x1(1), plot_x1(2), min(x(:,3))-2, max(x(:,3))+2]);
legend('Class 0', 'Class 1', ' Classifier (from scratch)', 'Classifier (fminsearch)');

% Plot cost function
figure(13); plot(cost);
title('Calculated Cost');
xlabel('Iteration number'); ylabel('Cost');


%% ========================= Test Classifier  N=1000 ========================== %%
% Coefficients for decision boundary line equation
coeff(1,:) = polyfit([plot_x1(1), plot_x1(2)], [plot_x2(1,1), plot_x2(1,2)], 1);
coeff(2,:) = polyfit([plot_x1(1), plot_x1(2)], [plot_x2(2,1), plot_x2(2,2)], 1); %fminsearch
% Decide based on which side of the line each point is on
for i = 1:2
    if coeff(i,1) >= 0
        decision_p2(:,i) = (coeff(i,1).*x_test(:,1) + coeff(i,2)) < x_test(:,2);
    else
        decision_p2(:,i) = (coeff(i,1).*x_test(:,1) + coeff(i,2)) > x_test(:,2);
    end
end
% 
% error1 = plot_test_data(decision_p2(:,1), label_test, Nc_test, p, 14, x_test, plot_x1, plot_x2(1,:));
% title('Test Data Classification (from scratch)');
% fprintf('Total error (classifier from scratch): %.2f%%\n',error1);

error2 = plot_test_data(decision_p2(:,2), label_test, Nc_test, p, 15, x_test, plot_x1, plot_x2(2,:));
title('Test Data Classification (using fminsearch)');
fprintf('Total error (classifier using fminsearch): %.2f%%\n',error2);

%% ================== Generate and Plot Training Set N=10 p3================== %%

N = 10;   % number of iid samples

% Generating true class labels
label = (rand(1,N) >= p(1))';
Nc = [length(find(label==0)),length(find(label==1))];

% Draw samples from each class pdf
x = zeros(N,n); 
for L = 0:1
    x(label==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc(L+1));
end

%Plot samples with true class labels
figure(16);
plot(x(label==0,1),x(label==0,2),'o',x(label==1,1),x(label==1,2),'+');
legend('Class 0','Class 1'); title('Training Data and True Class Labels');
xlabel('x_1'); ylabel('x_2'); hold on;

%% ======================== Logistic Regression  N=10 p3======================= %%
% Initialize fitting parameters
xog = [ones(N, 1) x];
xq = [ones(N, 1) x(:, 1) x(:, 2) x(:, 1).^2 x(:, 1).*x(:, 2) x(:, 2).^2];

initial_theta = zeros(n+1, 1);
initial_thetaq = zeros(6, 1);


label=double(label);

% Compute gradient descent to get theta values
[theta1, cost1] = fminsearch(@(t)(cost_func(t, xog, label, N)), initial_theta);
newtheta = [theta1;0;0;0];
[theta2, cost2] = fminsearch(@(t)(cost_func(t, xq, label, N)), newtheta);
decision_boundary = @(x1,x2) theta2(1) + theta2(2)*x1 + theta2(3)*x2 + theta2(4)*x1*x1 + theta2(5)*x1*x2 + theta2(6)*x2*x2;


hold on;
fimplicit(decision_boundary, [-9, 9], 'DisplayName', 'Decision boundary');
xlabel('x_1'), ylabel('x_2');
title('logistic quadratic function model');
legend();
axis equal;
hold off;

%% ================== Generate and Plot Training Set N=100 p3================== %%

N = 100;   % number of iid samples

% Generating true class labels
label = (rand(1,N) >= p(1))';
Nc = [length(find(label==0)),length(find(label==1))];

% Draw samples from each class pdf
x = zeros(N,n); 
for L = 0:1
    x(label==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc(L+1));
end

%Plot samples with true class labels
figure(17);
plot(x(label==0,1),x(label==0,2),'o',x(label==1,1),x(label==1,2),'+');
legend('Class 0','Class 1'); title('Training Data and True Class Labels');
xlabel('x_1'); ylabel('x_2'); hold on;

%% ======================== Logistic Regression  N=100 p3======================= %%
% Initialize fitting parameters
xog = [ones(N, 1) x];
xq = [ones(N, 1) x(:, 1) x(:, 2) x(:, 1).^2 x(:, 1).*x(:, 2) x(:, 2).^2];

initial_theta = zeros(n+1, 1);
initial_thetaq = zeros(6, 1);


label=double(label);

% Compute gradient descent to get theta values
[theta1, cost1] = fminsearch(@(t)(cost_func(t, xog, label, N)), initial_theta);
newtheta = [theta1;0;0;0];
[theta2, cost2] = fminsearch(@(t)(cost_func(t, xq, label, N)), newtheta);
decision_boundary = @(x1,x2) theta2(1) + theta2(2)*x1 + theta2(3)*x2 + theta2(4)*x1*x1 + theta2(5)*x1*x2 + theta2(6)*x2*x2;


hold on;
fimplicit(decision_boundary, [-9, 9], 'DisplayName', 'Decision boundary');
xlabel('x_1'), ylabel('x_2');
title('logistic quadratic function model');
legend();
axis equal;
hold off;

%% ================== Generate and Plot Training Set N=1000 p3================== %%

N = 1000;   % number of iid samples

% Generating true class labels
label = (rand(1,N) >= p(1))';
Nc = [length(find(label==0)),length(find(label==1))];

% Draw samples from each class pdf
x = zeros(N,n); 
for L = 0:1
    x(label==L,:) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc(L+1));
end

%Plot samples with true class labels
figure(18);
plot(x(label==0,1),x(label==0,2),'o',x(label==1,1),x(label==1,2),'+');
legend('Class 0','Class 1'); title('Training Data and True Class Labels');
xlabel('x_1'); ylabel('x_2'); hold on;

%% ======================== Logistic Regression  N=1000 p3======================= %%
% Initialize fitting parameters
xog = [ones(N, 1) x];
xq = [ones(N, 1) x(:, 1) x(:, 2) x(:, 1).^2 x(:, 1).*x(:, 2) x(:, 2).^2];

initial_theta = zeros(n+1, 1);
initial_thetaq = zeros(6, 1);


label=double(label);

% Compute gradient descent to get theta values
[theta1, cost1] = fminsearch(@(t)(cost_func(t, xog, label, N)), initial_theta);
newtheta = [theta1;0;0;0];
[theta2, cost2] = fminsearch(@(t)(cost_func(t, xq, label, N)), newtheta);
decision_boundary = @(x1,x2) theta2(1) + theta2(2)*x1 + theta2(3)*x2 + theta2(4)*x1*x1 + theta2(5)*x1*x2 + theta2(6)*x2*x2;


hold on;
fimplicit(decision_boundary, [-9, 9], 'DisplayName', 'Decision boundary');
xlabel('x_1'), ylabel('x_2');
title('logistic quadratic function model');
legend();
axis equal;
hold off;

%% ============================ Functions ============================= %%
function [theta, cost] = gradient_descent(x, N, label, theta, alpha, num_iters)
    cost = zeros(num_iters, 1);
    for i = 1:num_iters % while norm(cost_gradient) > threshold
        h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function   
        cost(i) = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))));
        cost_gradient = (1/N)*(x' * (h - label));
        theta = theta - (alpha.*cost_gradient); % Update theta
    end
end

function cost = cost_func(theta, x, label,N)
    h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function
    cost = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))));
end

function error = plot_test_data(decision, label, Nc, p, fig, x, plot_x1, plot_x2)
    ind00 = find(decision==0 & label==0); % true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % false negative
    ind11 = find(decision==1 & label==1); % true positive
    error = (p10*p(1) + p01*p(2))*100;

    % Plot decisions and decision boundary
    figure(fig);
    plot(x(ind00,1),x(ind00,2),'og'); hold on,
    plot(x(ind10,1),x(ind10,2),'or'); hold on,
    plot(x(ind01,1),x(ind01,2),'+r'); hold on,
    plot(x(ind11,1),x(ind11,2),'+g'); hold on,
    plot(plot_x1, plot_x2);
    axis([plot_x1(1), plot_x1(2), min(x(:,2))-2, max(x(:,2))+2])
    legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions','Classifier');
end
