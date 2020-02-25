function HW1



function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean m?mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);