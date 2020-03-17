%% Lorenz 63 system dynamics
clc;
close all;
clear all;
addpath('./regtools','./utils')
rng(100);

%% Dynamical system parameters
%System parameters
sig = 10;
rho = 28;
beta = 8/3;
param = [sig,rho,beta];
n = 3; %# state variables;

%Number of samples after trimming the ends
sampleTime = 0.01; %Sampling time
m = 200; %Number of samples after trimming

%Time parameters
dt = 0.0001;
t0 = 0;
tf = t0 + m*sampleTime;

%Intial conditions
x0 = [-8;7;27];

tspanf = t0:dt:(tf-dt);

%State sapace model of a 2nd order linear differential equation
options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
[xf] = ode5(@(t,x) LorentzSys63(t,x,param),tspanf,x0');
x = xf;
t = tspanf';

%% Sample dynamical system
dsample = sampleTime/dt;
timeData = t(1:dsample:end);
Nsamples = length(timeData);

%% True state variables
x1true = x(1:dsample:end,1);
x2true = x(1:dsample:end,2);
x3true = x(1:dsample:end,3);
xDataT = [x1true,x2true,x3true];

%% True state derivatives
dx1true = sig*(x2true-x1true);
dx2true = x1true.*(rho - x3true) - x2true;
dx3true = x1true.*x2true - beta*x3true;
dxDataT = [dx1true,dx2true,dx3true];

%% Noisy state derivatives
dxDataN = zeros(Nsamples,n);

%% NOISE MODELS

noise = 0.01;
sigma1 = noise;
sigma2 = noise;
sigma3 = noise;
%     
SNR1 = (rms(x1true)/sigma1)^2;
SNR2 = (rms(x2true)/sigma2)^2;
SNR3 = (rms(x3true)/sigma3)^2;

x1noisy = x1true + sigma1*randn(Nsamples,1);

x2noisy = x2true + sigma2*randn(Nsamples,1);

x3noisy = x3true + sigma3*randn(Nsamples,1);

xDataN = [x1noisy,x2noisy,x3noisy];

errorxsn(1) = norm(x1noisy - x1true)/norm(x1true);
errorxsn(2) = norm(x2noisy - x2true)/norm(x2true);
errorxsn(3) = norm(x3noisy - x3true)/norm(x3true);

errorxs(1) = norm(x1noisy - x1true);
errorxs(2) = norm(x2noisy - x2true);
errorxs(3) = norm(x3noisy - x3true);

xi = xDataN';
%% GENERATE BASIS MATRIX
deg = 3;

%% Generate multi index array (graded lexicographic order)
%Dimension of the basis
dim = nchoosek(n + deg,deg);
%Construct the multi index array

if n == 1
    alpha(:,1) = 0:1:deg;
else
    alpha = nd_multi_index_poly(n,deg);
end

%Generate multidimensional, non-orthogonal polynomial basis matrix
X = zeros(m,dim);
for j = 1:dim
            alphajs = repmat(alpha(j,:),m,1);
            X(:,j) = prod(xi'.^(alphajs),2);
end

%% ORTHOGONALIZATION WITH RESPECT TO THE SAMPLE MEASURE 
[Qmgs,Rmgs] = MGS_nD_samples(X);
G_MGS = 1/m*Qmgs'*Qmgs;

%Check orthogonality (Mutual coherence)
A = zeros(dim);
for c = 1:dim
    A(:,c) = G_MGS(:,c)/norm(G_MGS(:,c));
end
MC = max(max(abs(A - diag(diag(A)))))
%Check orthonormality
e_OrthN = norm(G_MGS - eye(dim),'f')

%Compute condition number
fprintf('Condition number of the polynomial basis:')
cond(X)
fprintf('Condition number of the orthogonal basis:')
cond(Qmgs)
