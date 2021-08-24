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
sigma = 0*0.1;
E = sigma * randn(m_samples,n_dim); % Additive zero-mean white noise

X = X_exact + E; 
X0 = repmat(X(1,:),m_samples,1);
X = X(:);
X0 = X0(:);
t_data = t_span;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Spline basis setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
spl_deg = 3;%Polynomial degree
M = spl_deg + 1;%Spline order
t_half = 0.5*(t_data(1:end-1) + t_data(2:end));
% knot_vec = [t_data(1),t_data(1),t_data(1),t_data(1),t_data,t_data(end),t_data(end),t_data(end),t_data(end)];
knot_vec = [t_data(1),t_data(1),t_data(1),t_data,t_data(end),t_data(end),t_data(end)];

[B,~] = bspline_basismatrix(M,knot_vec,t_data);
[B_half,~] = bspline_basismatrix(M,knot_vec,t_half);
B_big = blkdiag(B,B);
B_big_half = blkdiag(B_half,B_half);
n_ctrpts = size(B,2);%# control points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Reference parameters
%B-spline parameters (solve spline model with exact measurements)
Q = B_big' * X_exact(:);
theta_ref = (B_big'*B_big)\Q;

%Check spline model
X_spline = B_big*theta_ref;
X_spline = reshape(X_spline,[],n_dim);
figure;
plot(t_data,X_exact,'Marker','.');
hold on;
plot(t_data,X_spline,'linestyle','--');
xlabel('Time')
ylabel('state variable X')
legend('exact x_1','exact x_2','spline x_1','spline x_2')
title('Spline model with exact measurements');
pause;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Parameter initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pert = 10;
theta0 = theta_ref + pert*rand(n_ctrpts * n_dim,1);
n_theta = length(theta0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Levenberg-Marquardt setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_iter = 50;
Ldown = 9;
Lup = 11;
reg_lm = 1e-2;
adaptive_LM = 1;
tol_update = 1e-3;

fd_pert = 0.0001;
grad_fd = zeros(n_theta,1);
Jac_fd = zeros(n_dim*m_samples,n_theta);

theta = zeros(n_theta,n_iter);
%Loss
loss = zeros(n_iter,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LEVENBERG-MARQUARDT OPTIMIZER

%Start optimization procedure
for k = 1:n_iter
    
    %Extract paramters
    theta_k = theta(:,k);
    
    %Spline model
    Y = B_big * theta_k;
    Y_half = B_big_half * theta_k;
    y_half = reshape(Y_half,m_samples-1,n_dim);
    y0 = Y([1,m_samples+1])'; Y0 = repmat(y0,m_samples,1); Y0 = Y0(:);
    Y_tilde = Y - Y0; Y_tilde([1,m_samples+1]) = [];
    B_big_tilde = B_big;
    B_big_tilde([1,m_samples+1],:) = [];
    
    %Compute residual and loss
    res_theta = X - B_big * theta_k;    
    loss(k) = 0.5*res_theta'*res_theta;
    
    %Compute Jacobian
    
    Jac = -B_big;
    
    %% Check gradient and Jacobian using finite differences
    
    for i=1:n_theta
        
        pert_vec = zeros(n_theta,1);
        pert_vec(i) = fd_pert;

        %Compute residual and loss
        theta_minus = theta(:,k) - pert_vec;
        res_theta_minus = X - B_big * theta_minus;        
        loss_minus = 0.5*res_theta_minus' * res_theta_minus;
        
        %Compute residual and loss
        theta_plus = theta(:,k) + pert_vec;
        res_theta_plus = X - B_big * theta_plus;        
        loss_plus = 0.5*res_theta_plus' * res_theta_plus;

        grad_fd(i) = (loss_plus - loss_minus) / (2 * fd_pert);
        Jac_fd(:,i) = (res_theta_plus - res_theta_minus) / (2 * fd_pert);
        
    end
    
    %% Solve for optimal direction
    %Compute increment
    Hess = Jac'*Jac;
    grad = Jac'*res_theta;
    Hess_fd = Jac_fd'*Jac_fd;
    
%     [grad,grad_fd]
%     
%     Jac_diff = Jac - Jac_fd;
%     Jac_diff(abs(Jac_diff) < 1e-6) = 0; 
%     figure
%     subplot(1,3,1);
%     spy(Jac_fd)
%     subtitle('Finite Differences Jacobian')
%     subplot(1,3,2);
%     spy(Jac)
%     subtitle('Analytical Jacobian')
%     subplot(1,3,3);
%     spy(Jac_diff)
%     subtitle('Jacobian difference')
%     
%     pause
%     close
    
    LHS = Hess + reg_lm*diag(diag(Hess));
    RHS = -grad;
    hlm = LHS\RHS;
    
%     error_grad = norm(grad - grad_fd);
%     error_Jac = norm(Jac - Jac_fd);
%     fprintf('gradient error compared to FD is: %0.8f\n', error_grad);
%     fprintf('Jacobian error compared to FD is: %0.8f', error_Jac);
%     pause
    
    %% Adaptive LM
    loss_old = loss(k);        
    theta_try = theta(:,k) + hlm;
    %Compute new loss
    res_theta_try = X - B_big * theta_try;
    loss_try = 0.5*res_theta_try'*res_theta_try;
    
    %Compute adaptive rho parameter in LM (reference: The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems)
    num = loss_old - loss_try;
%     den = hlm'*((reg_lm*diag(diag(Hess)))*hlm - grad);
    den = hlm'*((reg_lm*diag(diag(Hess)))*hlm - grad);
    %         den = hlm'*(reg_lm*hlm + r);
    rho = num/den;

    if rho > tol_update
        theta(:,k+1) = theta_try;
        reg_lm = max(reg_lm/Ldown,1e-7);
        fprintf('Step accepted! rho = %0.6f \n\n',rho)
    else
        theta(:,k+1) = theta(:,k);
        reg_lm = min(reg_lm*Lup,1e7);
        fprintf('Step rejected! rho = %0.6f \n\n',rho)
    end
    
end

Theta_rec = reshape(theta(:,end),[],n_dim);
theta_rec = Theta_rec(:);

%% Model verification
%Comparison solution using LS and LM
Q = B_big' * X;
theta_ls = (B_big'*B_big)\Q;

LM_LS_diff = norm(theta_ls - theta_rec);
fprintf('Norm of difference between LM adn LS solutions: %0.8f',LM_LS_diff);

figure;
%Spline model 
Y_rec = B_big * theta_rec;
Y_rec = reshape(Y_rec,[],n_dim);
Y_ls = B_big * theta_ls;
Y_ls = reshape(Y_ls,[],n_dim);
plot(t_data,X_exact,'Marker','.', 'Color','g');
hold on;
X = reshape(X,[],n_dim);
plot(t_data,X,'Marker','.', 'Color','k');
plot(t_data,Y_ls,'linestyle','-.', 'Color','c');
plot(t_data,Y_rec,'linestyle','--', 'Color','r');

xlabel('Time')
ylabel('state variable X')
legend('exact x_1','exact x_2', 'data x_1','data x_2','OLS spline x_1','OLS spline x_2', 'LM spline x_1','LM spline x_2')
title('Spline model with recovered measurements');