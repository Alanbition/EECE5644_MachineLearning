function x = exam4q1_generateData(N)
close all,
m(:,1) = [-9;-4]; Sigma(:,:,1) = 4*[1,0.8;0.8,1]; % mean and covariance of data pdf conditioned on label 3
m(:,2) = [0;0]; Sigma(:,:,2) = 3*[3,0;0,0.3]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [8;-3]; Sigma(:,:,3) = 5*[1,-0.9;-0.9,1]; % mean and covariance of data pdf conditioned on label 1
componentPriors = [0.3,0.5,0.2]; thr = [0,cumsum(componentPriors)];
%N = 1000; 
u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
%figure(1),clf, %colorList = 'rbg';
for l = 1:3
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
%    figure(1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
end
figure(1), plot(x(1,:),x(2,:),'.'),
xlabel('X_1'); ylabel('X_2');
