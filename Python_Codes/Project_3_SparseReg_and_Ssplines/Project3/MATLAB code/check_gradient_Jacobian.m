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
sigma = 0.0001;
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
knot_vec = [t_data(1), t_data(1), t_data(1), t_data, t_data(end), t_data(end), t_data(end)]; %repeat knots at the boundaries 4 times per side
[B,~] = bspline_basismatrix(M,knot_vec,t_data);
[B_half,~] = bspline_basismatrix(M,knot_vec,t_half);
B_big = blkdiag(B,B);
B_big_half = blkdiag(B_half,B_half);
n_ctrpts = size(B,2);%# control points
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

%B-spline parameters (solve spline model with exact measurements)
Q = B_big' * X;
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
pert = 1;
xi0 = xi_true + pert*rand(n_basis * n_dim,1);
theta0 = theta_ref + pert*rand(n_ctrpts * n_dim,1);
n_xi = length(xi0);
n_theta = length(theta0);

zeta0 = [xi0;theta0];
n_zeta = length(zeta0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Levenberg-Marquardt setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_iter = 100;
Ldown = 9;
Lup = 11;
reg_lm = 1e-2;
adaptive_LM = 1;
tol_update = 1e-3;

% Regularization parameter
lambda = 0.001;
eps_w = 0.0001;
p_w = 2;

alpha = 0.1;

%Parameters
zeta = zeros(length(zeta0),n_iter);
zeta(:,1) = zeta0;

fd_pert = 0.000001;
grad_fd = zeros(n_xi + n_theta,1);
Jac_fd = zeros(n_dim*(2*m_samples - 1) + n_xi ,n_xi + n_theta);


%Loss
loss = zeros(n_iter,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% LEVENBERG-MARQUARDT OPTIMIZER

%Start optimization procedure
for k = 1:n_iter
    
    %Extract paramters
    xi_k = zeta(1:n_xi,k);
    theta_k = zeta(n_xi + 1:end,k);
    %Compute solution dependent weights
    w = xi_k;
    W = diag(1./(abs(w).^(p_w+1) + eps_w));
    L = sqrt(W);
    
    %Spline model
    Y = B_big * theta_k;
%     Y = X_exact(:);
    Y_half = B_big_half * theta_k;
%     Y_half = X_exact_half(:);
    y_half = reshape(Y_half,m_samples-1,n_dim);
    y0 = Y([1,m_samples+1])'; Y0 = repmat(y0,m_samples,1); Y0 = Y0(:);
    Y_tilde = Y - Y0; Y_tilde([1,m_samples+1]) = [];
    B_big_tilde = B_big;
    B_big_tilde([1,m_samples+1],:) = [];%remove first sample for each dimension
    
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
    res_theta = X - B_big * theta_k;
    res_l1 = L*xi_k;
    
    res = [res_xi ; sqrt(alpha)*res_theta ; sqrt(lambda)*res_l1];
    
    loss(k) = 0.5*res'*res;
    fprintf('loss = %0.6f \n\n',loss(k))
    %Compute Jacobian
    
%     j12 = zeros(n_dim,n_ctrpts * n_dim);
    J12 = zeros(size(B_big_tilde));
    temp = zeros(n_basis,n_dim);
    Xi = reshape(xi_k, n_basis, n_dim);
    
%     for i = 1:m_samples-1
%         Dphi_Dy = dphi_dx(y_half(i,:));
%         %Xi = reshape(xi_k, n_basis, n_dim);
%         M = Xi'*Dphi_Dy;
%         Me = repelem(M,1,n_ctrpts);
%         Be = repmat(B(i+1,:),n_dim,n_dim);%from t_1 not t_0
%         
%         j12 = j12 + dt*Me.*Be;
%         idx = i + (m_samples-1)*((1:n_dim) -1);
%         J12(idx,:) = j12;
%     end
    
    for i = 1:m_samples-1
       
        for q = 1:i
            temp = temp + dt*dphi_dx(y_half(i,:));
        end
        M = Xi'*temp;
        Me = repelem(M,1,n_ctrpts);
        Be = repmat(B_half(i,:),n_dim,n_dim);
        idx = i + (m_samples-1)*((1:n_dim) -1);
        J12(idx,:) = Me.*Be;
        temp = 0*temp;
    end
    
    J12 =  B_big_tilde - J12;
    
    Jac = [-D_big, J12;...
           zeros(n_dim * m_samples,n_xi), -sqrt(alpha)*B_big;...
           sqrt(lambda)*L, zeros(n_xi,n_theta)];
    
    %% Check gradient and Jacobian using finite differences

    for l=1:n_zeta
        
        pert_vec = zeros(n_zeta,1);
        pert_vec(l) = fd_pert;
        
        %Compute residual and loss
        zeta_minus = zeta(:,k) - pert_vec;
        %Extract paramters
        xi_k = zeta_minus(1:n_xi);
        theta_k = zeta_minus(n_xi + 1:end);
        %Compute solution dependent weights
        w = xi_k;
        W = diag(1./(abs(w).^(p_w+1) + eps_w));
        L = sqrt(W);

        %Spline model
        Y = B_big * theta_k;
        Y_half = B_big_half * theta_k;
        y_half = reshape(Y_half,m_samples-1,n_dim);
        y0 = Y([1,m_samples+1])'; Y0 = repmat(y0,m_samples,1); Y0 = Y0(:);
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
        
        res_xi_minus = Y_tilde - D_big * xi_k;
        res_theta_minus = X - B_big * theta_k;
        res_l1_minus = L*xi_k;
        res_minus = [res_xi_minus ; sqrt(alpha)*res_theta_minus ; sqrt(lambda)*res_l1_minus];
        
        loss_minus = 0.5*res_minus' * res_minus;
        
        %Compute residual and loss
        zeta_plus = zeta(:,k) + pert_vec;
        
        %Extract paramters
        xi_k = zeta_plus(1:n_xi);
        theta_k = zeta_plus(n_xi + 1:end);
        %Compute solution dependent weights
        w = xi_k;
        W = diag(1./(abs(w).^(p_w+1) + eps_w));
        L = sqrt(W);

        %Spline model
        Y = B_big * theta_k;
        Y_half = B_big_half * theta_k;
        y_half = reshape(Y_half,m_samples-1,n_dim);
        y0 = Y([1,m_samples+1])'; Y0 = repmat(y0,m_samples,1); Y0 = Y0(:);
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
        
        res_xi_plus = Y_tilde - D_big * xi_k;
        res_theta_plus = X - B_big * theta_k;
        res_l1_plus = L*xi_k;
        res_plus = [res_xi_plus ; sqrt(alpha)*res_theta_plus ; sqrt(lambda)*res_l1_plus];
        
        loss_plus = 0.5*res_plus' * res_plus;
        
        grad_fd(l) = (loss_plus - loss_minus) / (2 * fd_pert);
        Jac_fd(:,l) = (res_plus - res_minus) / (2 * fd_pert);

    end
    
    
    %% Solve for optimal direction
    %Compute increment
    Hess = Jac'*Jac;
    grad = Jac'*res;
    Hess_fd = Jac_fd'*Jac_fd;
    
%     [grad,grad_fd]
%     
    Jac_diff = Jac - Jac_fd;
    Jac_diff(abs(Jac_diff) < 1e-6) = 0; 
    figure
    subplot(1,3,1);
    spy(Jac_fd)
    subtitle('Finite Differences Jacobian')
    subplot(1,3,2);
    spy(Jac)
    subtitle('Analytical Jacobian')
    subplot(1,3,3);
    spy(Jac_diff)
    subtitle('Jacobian difference')
    
    pause
    close
    
    LHS = Hess_fd + reg_lm*diag(diag(Hess_fd));
    RHS = -grad_fd;
    hlm = LHS\RHS;
    
%     error_grad = norm(grad - grad_fd);
%     error_Jac = norm(Jac - Jac_fd);
%     fprintf('gradient error compared to FD is: %0.8f\n', error_grad);
%     fprintf('Jacobian error compared to FD is: %0.8f', error_Jac);
%     pause
    
    %% Adaptive LM
    loss_old = loss(k);        
    zeta_try = zeta(:,k) + hlm;
    xi_try = zeta_try(1:n_xi);
    theta_try = zeta_try(n_xi+1:end);
    %Compute new loss
    res_xi_try = Y_tilde - D_big * xi_try;
    res_theta_try = X - B_big * theta_try;
    res_l1_try = L*xi_try;
    
    res_try = [res_xi_try ; sqrt(alpha)*res_theta_try ; sqrt(lambda)*res_l1_try];
    loss_try = 0.5*res_try'*res_try;
    
    %Compute adaptive rho parameter in LM (reference: The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems)
    num = loss_old - loss_try;
%     den = hlm'*((reg_lm*diag(diag(Hess)))*hlm - grad);
    den = hlm'*((reg_lm*diag(diag(Hess_fd)))*hlm - grad_fd);
    %         den = hlm'*(reg_lm*hlm + r);
    rho = num/den;

    if rho > tol_update
        zeta(:,k+1) = zeta_try;
%         reg_lm = max(reg_lm/Ldown,1e-7);
        reg_lm = 1e3;

        fprintf('Step accepted! rho = %0.6f \n\n',rho)
    else
        zeta(:,k+1) = zeta(:,k);
%         reg_lm = min(reg_lm*Lup,1e7);
        reg_lm = 1e3;

        fprintf('Step rejected! rho = %0.6f \n\n',rho)
    end
    
end

Xi_rec = reshape(zeta(1:n_xi,end),[],n_dim)
Theta_rec = reshape(zeta(n_xi+1:end,end),[],n_dim)
xi_rec = Xi_rec(:);
theta_rec = Theta_rec(:);

%% Model verification

figure;

subplot(2,1,1);
%Spline model 
Y_rec = B_big * theta_rec;
Y_rec = reshape(Y_rec,[],n_dim);
plot(t_data,X_exact,'Marker','.');
hold on;
plot(t_data,Y_rec,'linestyle','--');
xlabel('Time')
ylabel('state variable X')
legend('exact x_1','exact x_2','spline x_1','spline x_2')
title('Spline model with recovered measurements');

subplot(2,1,2);
%Dynamical model

%Build dictionary and evaluate at Y
Phi = zeros(m_samples,n_basis);
for isim = 1:m_samples
    crow = piset_monomial(Y_rec(isim,:),index_pc);
    Phi(isim,:) = crow(1:n_basis);
end
T = dt*tril(ones(m_samples));
Psi = T*Phi;
D_big = blkdiag(Psi,Psi);

X_rec = X0 + D_big * xi_rec;
X_rec = reshape(X_rec,[],n_dim);
plot(t_data,X_exact,'Marker','.');
hold on;
plot(t_data,X_rec,'linestyle','--');
xlabel('Time')
ylabel('state variable X')
legend('exact x_1','exact x_2','recovered x_1','recovered x_2')
title('Recovered states');