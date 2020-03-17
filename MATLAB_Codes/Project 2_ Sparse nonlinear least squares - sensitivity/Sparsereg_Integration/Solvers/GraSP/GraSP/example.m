%% This example demonstrates how to use the GraSP script for a sparsity-constrained logistic regression problem
% DATE: August 2012
% AUTHOR: Sohail Bahmani
%% Set parameters

n = 1000; % dimensionality of the signals/parameters
m = 100; % the number of samples
s = 10; % the desired sparsity

%% Generate the design matrix, the sparse signal, and its associated labels vector

% Generate the sparse signal
x_star = zeros(n,1);
x_star(1:s) = randn(s,1);
x_star = x_star(randperm(n)); % an n-dimensional vector with s non-zero entries

% Generate measurement/design matrix
A = randn(n,m); % Each *column* corresponds to one measurement/sample

% Find probabilities in logistic model for each label
q = 1./(1+exp(-A'*x_star));

% Generate labels according to the logistic model
y = zeros(m,1);
for i = 1:m    
    y(i) = randsrc(1,1,[0 1; 1-q(i) q(i)]);   
end

%% Define the cost function
F = @(x,I)myLogistic(x,A(I,:),y);

%% Apply the GraSP algorithm

% Configure options of GraSP
options.HvFunc = @(v,x,I)myLogistic_Hv(v,x,A(I,:)); % Hessian-vector product
options.maxIter = 100; % max. number of iterations 
options.tolF = 1e-6; % progress tol.
options.tolG = 0.35; % gradient tol.

% Without any penalty or constraint
xGraSP_1 = GraSP(F,s,n,options);

% With L2 penalty
options.mu = -0.1;
xGraSP_2 = GraSP(F,s,n,options);

% With L2 constraint
options.mu = norm(x_star);
xGraSP_3 = GraSP(F,s,n,options);