%% Solve 1 dof linear/quadratic undamped, unforced, oscillator problem
clc;
close all;
clear all;
addpath('./Solvers','./utils')
rng(100);

%% Dynamical system parameters
%System parameters
% omegap = 2*pi;
omegap = 1;

% epsip = -0.5000;
epsip = 5;

xip = 0.05;
% xip = 0.05;

param = [omegap,epsip,xip];
n = 2; %# state variables;

%Number of samples after trimming the ends
sampleTime = 0.01; %Sampling time
m = 200; %Number of samples after trimming

%Time parameters
dt = 0.0001;
t0 = 0;
tf = t0 + m*sampleTime;

%Intial conditions
x0 = [0;1];

tspanf = t0:dt:(tf-dt);

%State sapace model of a 2nd order linear differential equation
options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
[xf] = ode5(@(t,x) DuffingODE(t,x,param),tspanf,x0');
x = xf;
t = tspanf';

%% Sample dynamical system
dsample = sampleTime/dt;
timeData = t(1:dsample:end);
Nsamples = length(timeData);

%% True state variables
x1true = x(1:dsample:end,1);
x2true = x(1:dsample:end,2);
xDataT = [x1true,x2true];

%% True state derivatives
dx1true = x2true;
dx2true = -2*omegap*xip.*x2true - omegap^2.*(x1true +epsip.*x1true.^3);
dxDataT = [dx1true,dx2true];
%% Noisy state derivatives

%% NOISE MODELS
sigma1 = 0.01;
sigma2 = 0.01;

x1noisy = x1true + sigma1*randn(Nsamples,1);
x2noisy = x2true + sigma2*randn(Nsamples,1);

dx1noisy = dx1true + sigma1*randn(Nsamples,1);
dx2noisy = dx2true + sigma2*randn(Nsamples,1);

xDataN = [x1noisy, x2noisy];
dxDataN = [dx1noisy, dx2noisy];

dxm = dxDataN;
%% Form basis matrix

% Setup the basis
n=2;
poldeg = 4;
index_pc = nD_polynomial_array(n,poldeg);
p = size(index_pc,1);

%%% Build basis matrix %%%
Phi = zeros(m,p);
for isim = 1:m
    crow = piset_monomial(xDataN(isim,:),index_pc);
    Phi(isim,:) = crow(1:p);
end

%Normalize columns
Phicol_norm = (sqrt(sum(Phi.*Phi,1)))';
Wn = diag(1./Phicol_norm); %Normalization matrix 
Phin = Phi * Wn; %Column-normalized basis matrix


%% Initial guess on parameter vector
XiT = zeros(p,n);
XiT(3,1) = 1;
XiT(3,2) = -2*omegap*xip;
XiT(2,2) = - omegap^2;
XiT(7,2) = - omegap^2*epsip;

a = -1;
b = 1;
sc = 0.1;
r = a + (b-a).*rand(p,n);
Xi0 = XiT + sc*r;
%% Initial conditions
L = 100000;
alpha = 1/L;
Niter = 10000;
Cpsi = zeros(Niter+1,n);
Xi = Xi0;
errXi = zeros(Niter+1,n);
residual = zeros(Niter+1,n);

for s = 1:n
Xis = Xi(:,s);
dxp(:,s) = Phi*Xis;
errXi(1,s) = norm(Xis - XiT(:,s));
ress = dxm(:,s) - dxp(:,s);%residual vector for state s
residual(1,s) = norm(ress);
gradLs = -(ress'*Phi); %Gradient of L for state s
Cpsi(1,s) = 1/(2*m)*ress'*ress;
end

sp = [1,3];%sparsity level

figure
suptitle('Measured vs predicted state derivative for the initial guess')
subplot(2,1,1)
plot(timeData,dxm(:,1),'r.')
hold on
plot(timeData,dxp(:,1),'b.')
xlabel('Time')
ylabel('state dx_1')
legend('Measurement','Prediction')

subplot(2,1,2)
plot(timeData,dxm(:,2),'r.')
hold on
plot(timeData,dxp(:,2),'b.')
xlabel('Time')
ylabel('state dx_2')
legend('Measurement','Prediction')

fprintf('Condition number of the basis matrix:')
cond(Phi)
pause
close all

for iter = 1:Niter %iteration loop
    
    for s = 1:n %loop over states

        %% Compute error and gradL for state s
        Xis = Xi(:,s); %Solution vector for state s
        dxps = Phi*Xis;
        cond(Phi)
        ress = dxm(:,s) - dxps;%residual vector for state s (residual = xdot - Theta*Xi)
        gradLs = -(ress'*Phi); %Gradient of L for state s

        Cps = 1/(2*m)*ress'*ress; %Cost function = Least squares term + L1 regularization term for state s

        Xis = Xis - alpha*gradLs';%alpha = 1/L is the step size
        [Xisort,isort]=sort(abs(Xis));%Sort the coefficients from lowest to highest magnitude
        Xisnew=zeros(p,1);
        sps = sp(s);%sparsity level for state s
        Xisnew(isort(p-sps+1:p))=Xis(isort(p-sps+1:p));
        Xis=Xisnew;
        dxps = Phi*Xis;
        %Compute error and cost function with updated Xis
        ress = dxm(:,s) - dxps;
        residual(iter+1,s) = norm(ress);    
        Cpsi(iter+1,s) = 1/(2*m)*ress'*(ress);
        fprintf('iter = %5d Objective value of state %5d = %5.10f\n',iter,s,Cpsi(iter+1,s));
        
        errXi(iter+1,s) = norm(Xis - XiT(:,s))/norm(XiT(:,s));
        fprintf('iter = %5d Error value of state%5d = %5.10f\n\n',iter,s,errXi(iter+1,s));
        Xi(:,s) = Xis;
    
    end
    
    
end

figure
subplot(2,3,1)
title('Cost function vs iteration #')
semilogy(1:Niter+1,Cpsi(:,1),'r')
xlabel('iteration #')
ylabel('Cost function')
legend('state x_1')
subplot(2,3,4)
semilogy(1:Niter+1,Cpsi(:,2),'r--')
xlabel('iteration #')
ylabel('Cost function')
legend('state_x_2')

subplot(2,3,2)
title('Relative solution error vs iteration #')
semilogy(1:Niter+1,errXi(:,1),'b')
xlabel('iteration #')
ylabel('Relative solution error')
legend('state x_1')
subplot(2,3,5)
semilogy(1:Niter+1,errXi(:,2),'b--')
xlabel('iteration #')
ylabel('Relative solution error')
legend('state_x_2')

subplot(2,3,3)
title('residual vs iteration #')
semilogy(1:Niter+1,residual(:,1),'g')
xlabel('iteration #')
ylabel('Residual')
legend('state x_1')
subplot(2,3,6)
semilogy(1:Niter+1,residual(:,2),'g--')
xlabel('iteration #')
ylabel('Residual')
legend('state_x_2')
 
   