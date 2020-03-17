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
m = length(timeData);

%% True state variables
x1true = x(1:dsample:end,1);
x2true = x(1:dsample:end,2);
xDataT = [x1true,x2true];

%% True state derivatives
dx1true = x2true;
dx2true = -2*omegap*xip.*x2true - omegap^2.*(x1true +epsip.*x1true.^3);
dxDataT = [dx1true,dx2true];
%% Noisy state derivatives
dxDataN = zeros(m,n);

%% NOISE MODELS
sigma1 = 0.01; %Noise level state 1
sigma2 = 0.01; %Noise level state 2

x1noisy = x1true + sigma1*randn(m,1);
x2noisy = x2true + sigma2*randn(m,1);

dx1noisy = dx1true + sigma1*randn(m,1);
dx2noisy = dx2true + sigma2*randn(m,1);

xDataN = [x1noisy, x2noisy];
dxDataN = [dx1noisy, dx2noisy];

xm = xDataN;
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
%True solution vector
XiT = zeros(p,n);
XiT(3,1) = 1;
XiT(3,2) = -2*omegap*xip;
XiT(2,2) = - omegap^2;
XiT(7,2) = - omegap^2*epsip;

%Intial guess (perturbed)
a = -1;
b = 1;
sc = 0.01;%scaling factor of the perturbation for the initial guess
r = a + (b-a).*rand(p,n);
Xi0 = XiT + sc*r;

Xi = Xi0;
%% Initial conditions
x0m = x0;
S0 = zeros(n,p);
s0 = S0(:);
xs0 = [x0;s0];
nds = length(xs0);
dxS = zeros(n,p+1,length(timeData));

%Step size(L) and maximum number of iterations
L = 100000;
alpha = 1/L;
Niter = 1000;

%Sparsity level (number of non-zero entries in the solution vector for each
%state)
sp = [1,3];%[state1 state2]

%Solution error and Cost function
errXi = zeros(Niter+1,n);
Cpsi = zeros(Niter+1,n);
residual = zeros(Niter+1,n);
%Compute solution error and const function for the initial guess
for s = 1:n
    Xis = Xi(:,s);
    options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,nds));
    [tdxs,dxs] = ode45(@(t,x)IntegrateModel_s(t,x,Xis,index_pc),timeData,xs0,options);
    dxs = dxs';
    for t = 1:size(dxs,2)
        dxS(:,:,t) = reshape(dxs(:,t),n,p+1);
        xp = squeeze(dxS(:,1,:))';
        Sp = dxS(:,2:end,:);
    end
    
    errXi(1,s) = norm(Xis - XiT(:,s))/norm(XiT(:,s)); 
    ress = xm(:,s) - xp(:,s);%residual vector for state s
    residual(1,s) = norm(ress);
    Cpsi(1,s) = 1/(2*m)*ress'*ress;
end


figure
title('Measured vs predicted state for the initial guess')

subplot(2,1,1)
plot(timeData,xm(:,1),'r.')
hold on
plot(timeData,xp(:,1),'b.')
xlabel('Time')
ylabel('state x_1')
legend('Measurement','Prediction')

subplot(2,1,2)
plot(timeData,xm(:,2),'r.')
hold on
plot(timeData,xp(:,2),'b.')
xlabel('Time')
ylabel('state x_2')
legend('Measurement','Prediction')

pause
close all
clc

params{1} = timeData;
params{2} = xs0;
params{3}= index_pc;
params{4} = xm(:,s);
params{5} = n;
params{6} = p;
pause
        
        %% Levenberg Marquadt Algorithm
        opts = optimoptions(@lsqnonlin,'SpecifyObjectiveGradient',true,'Algorithm','levenberg-marquardt');
        [Xis,resnorm,res,eflag,output2] = lsqnonlin(@(Xi)myfun(Xi,params),Xi0,[],[],opts);
        

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

function [Cps,Sps] = myfun(Xi,params)
        
        %Extract parameters
        timeData = params{1};
        xs0 = params{2};
        index_pc = params{3};
        xm = params{4};
        n = params{5};
        p = params{6};
        s = params{7};
        m = length(xm);
        nds = length(xs0);
        
        %% Integrate state and Sensitivity using guessed parameters
        options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,nds));
        [tdxs,dxs] = ode45(@(t,x)IntegrateModel(t,x,Xi,index_pc),timeData,xs0,options);
        dxs = dxs';
        for t = 1:size(dxs,2)
        dxS(:,:,t) = reshape(dxs(:,t),n,p+1);
        xp = squeeze(dxS(:,1,:))';
        Sp = dxS(:,2:end,:);
        end
        
        %% Compute error and gradL for state s
        Xis = Xi(:,s); %Solution vector for state s
        ress = xm(:,s) - xp(:,s);%residual vector for state s %residual vector for state s (residual = x_meas - x_pred)
        Sps = squeeze(Sp(s,:,:))'; %Sensitivity for state s
        Cps = 1/(2*m)*ress'*ress; %Cost function = Least squares term for state s
end