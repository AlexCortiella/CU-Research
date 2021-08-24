%% Test integral form

%% Simultaneous sparse regression and denoising via splines
%% (INEFFICIENT IMPLEMENTATION)
clc;
close all;
clear all;
rng(2)
addpath('./utils');
%% Generate dynamical system data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t0 = 0;
tf = 2; 
dt = 0.01;
t_span = t0:dt:tf;
t_span_half = 0.5*(t_span(1:end-1) + t_span(2:end));

m_samples = length(t_span);

n_dim = 2;
x0 = [1;0];

%True parameters
gamma = 0.1;
kappa = 1;
epsilon = 5;

params = [gamma, kappa, epsilon];

opts = odeset('RelTol',1e-12,'AbsTol',1e-12);
[t,X_exact] = ode45(@(t,x)duffing(t, x, params), t_span, x0, opts);
[t_half,X_exact_half] = ode45(@(t,x)duffing(t, x, params), t_span_half, x0, opts);

x1 = X_exact(:,1);
x2 = X_exact(:,2);

% Compute true derivatives
dX_exact = zeros(m_samples,n_dim);
dX_exact(:,1) = x2;
dX_exact(:,2) =  -gamma * x2 - kappa * x1 - epsilon * x1.^3;

% Corrupt states by adding noise --> Observation model y(t) = x(t) + e(t)
sigma = 0.00000001;
E = sigma * randn(m_samples,n_dim); % Additive zero-mean white noise

X = X_exact + E; 
X_half = X_exact_half + E(1:end-1,:); 

X0 = repmat(X(1,:),m_samples,1);
X = X(:);
X_half = X_half(:);
X0 = X0(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Dynamics basis setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dyn_deg = 4;
index_pc = nD_polynomial_array(n_dim,dyn_deg);
n_basis = size(index_pc,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Reference parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Dynamical parameters
Xi_true = zeros(n_basis,n_dim);
Xi_true(3,1) = 1;
Xi_true(2,2) = - kappa;
Xi_true(3,2) = - gamma;
Xi_true(7,2) = - epsilon;

xi_true = Xi_true(:);

%% Parameter initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pert = 1;
xi0 = xi_true + pert*rand(n_basis * n_dim,1);
n_xi = length(xi0);

%% Levenberg-Marquardt setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_iter = 200;
Ldown = 9;
Lup = 11;
reg_lm = 1e-2;
adaptive_LM = 1;
tol_update = 1e-3;

% Regularization parameter
lambda = 0.001;
eps_w = 0.0001;
p_w = 2;

%Parameters
xi = zeros(length(xi0),n_iter);
xi(:,1) = xi0;

%Loss
loss = zeros(n_iter,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% LEVENBERG-MARQUARDT OPTIMIZER

%Start optimization procedure
for k = 1:n_iter
    
    %Extract paramters
    xi_k = xi(1:n_xi,k);
    %Compute solution dependent weights
    w = xi_k;
    W = diag(1./(abs(w) + eps_w));
    L = sqrt(W);
    
    Y_half = X_half(:);
    y_half = reshape(Y_half,m_samples-1,n_dim);
    y0 = X([1,m_samples+1])'; Y0 = repmat(y0,m_samples,1); Y0 = Y0(:);
    Y = X;
    Y_tilde = Y - Y0; Y_tilde([1,m_samples+1]) = [];
    
    %Build dictionary and evaluate at Y
    Phi = zeros(m_samples-1,n_basis);
    for isim = 1:m_samples-1
        crow = piset_monomial(y_half(isim,:),index_pc);
        Phi(isim,:) = crow(1:n_basis);
    end
    T = dt*tril(ones(m_samples-1));
    Psi = T*Phi;
    D_big = blkdiag(Psi,Psi);
    
    %Compute residual and loss
    res_xi = Y_tilde - D_big * xi_k;
    res_l1 = L*xi_k;
    
    res = [res_xi ; sqrt(lambda)*res_l1];
    
    loss(k) = res'*res;
    
    %Compute Jacobian
    Xi = reshape(xi_k, n_basis, n_dim);
        
    Jac = [-D_big;...
           sqrt(lambda)*L];
    
    %% Solve for optimal direction
    %Compute increment
    Hess = Jac'*Jac;
    grad = Jac'*res;
    LHS = Hess + reg_lm*diag(diag(Hess));
    RHS = -grad;
    hlm = LHS\RHS;

    %% Adaptive LM
    loss_old = loss(k);        
    xi_try = xi(:,k) + hlm;
    %Compute new loss
    res_xi_try = Y_tilde - D_big * xi_try;
    res_l1_try = L*xi_try;
    
    res_try = [res_xi_try ; sqrt(lambda)*res_l1_try];
    loss_try = res_try'*res_try;
    
    %Compute adaptive rho parameter in LM (reference: The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems)
    num = loss_old - loss_try;
    den = hlm'*((reg_lm*diag(diag(Hess)))*hlm - grad);
    %         den = hlm'*(reg_lm*hlm + r);
    rho = num/den;

    if rho > tol_update
        xi(:,k+1) = xi_try;
        reg_lm = max(reg_lm/Ldown,1e-7);
        fprintf('Step accepted! rho = %0.6f \n\n',rho)
    else
        xi(:,k+1) = xi(:,k);
        reg_lm = min(reg_lm*Lup,1e7);
        fprintf('Step rejected! rho = %0.6f \n\n',rho)
    end
    
end

Xi_rec = reshape(xi(1:n_xi,end),[],n_dim);
xi_rec = Xi_rec(:);

%% Model verification
%Dynamical model

%% OLS and WBPDN solutions

Y_half = X_half(:);
y_half = reshape(Y_half,m_samples-1,n_dim);
y0 = X([1,m_samples+1])'; Y0 = repmat(y0,m_samples,1); Y0 = Y0(:);
Y_tilde = Y - Y0; Y_tilde([1,m_samples+1]) = [];

%Build dictionary and evaluate at Y
Phi = zeros(m_samples-1,n_basis);
for isim = 1:m_samples-1
    crow = piset_monomial(y_half(isim,:),index_pc);
    Phi(isim,:) = crow(1:n_basis);
end
T = dt*tril(ones(m_samples-1));
Psi = T*Phi;
D_big = blkdiag(Psi,Psi);

%OLS solution
xi_ls = D_big\Y_tilde;
Xi_ls = reshape(xi_ls,[],n_dim);

Y

%WBPDN solution
num_itr = 3;

for itr = 1:num_itr+1

    if itr == 1

        [xi_wbpdn, residual, reg_residual] = solveWBPDN(D_big, Y_tilde, [], lambda, n_basis * n_dim);

    else

        %Apply weighting
        w = 1 ./ (abs(xi_wbpdn).^p_w + eps_w);

        [xi_wbpdn, residual, reg_residual] = solveWBPDN(D_big, Y_tilde, w, lambda, n_basis * n_dim);

    end

end

Xi_wbpdn = reshape(xi_wbpdn,[],n_dim);

fprintf('Recovered coefficient vectors: \n');
[Xi_rec, Xi_ls, Xi_wbpdn]

error_lm = norm(Xi_rec - Xi_true)/norm(Xi_true);
error_ls = norm(Xi_ls - Xi_true)/norm(Xi_true);
error_wbpdn = norm(Xi_wbpdn - Xi_true)/norm(Xi_true);

fprintf('Relative coefficient errors: \n')
error_comp = [error_lm, error_ls, error_wbpdn]

fprintf('Residuals: \n')
residual_lm = norm(Y_tilde - D_big * xi_rec);
residual_ls = norm(Y_tilde - D_big * xi_ls);
residual_wbpdn = norm(Y_tilde - D_big * xi_wbpdn);
residual_comp = [residual_lm, residual_ls, residual_wbpdn]

%% Plots

%Build dictionary and evaluate at Y
Phi = zeros(m_samples,n_basis);
X2 = reshape(X,[],n_dim);
for isim = 1:m_samples
    crow = piset_monomial(X2(isim,:),index_pc);
    Phi(isim,:) = crow(1:n_basis);
end
T = dt*tril(ones(m_samples));
Psi = T*Phi;
D_big = blkdiag(Psi,Psi);

X_rec = X0 + D_big * xi_rec;
X_ls = X0 + D_big * xi_ls;
X_wbpdn = X0 + D_big * xi_wbpdn;

X_rec = reshape(X_rec,[],n_dim);
X_ls = reshape(X_ls,[],n_dim);
X_wbpdn = reshape(X_wbpdn,[],n_dim);


plot(t,X_exact,'Marker','.');
hold on;
plot(t,X_rec,'linestyle','--');
plot(t,X_ls,'linestyle','-.');
plot(t,X_wbpdn,'linestyle',':');

xlabel('Time')
ylabel('state variable X')
legend('exact x_1','exact x_2','LM x_1','LM x_2',...
       'OLS x_1','OLS x_2', 'WBPDN x_1','WBPDN x_2')
title('Recovered states');


