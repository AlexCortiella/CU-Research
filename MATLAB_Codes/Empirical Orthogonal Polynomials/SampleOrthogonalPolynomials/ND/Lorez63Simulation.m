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


sampleTime = 10; %Sampling time
m = 1e6; %Number of samples after trimming

%Time parameters
t0 = 0;
tf = 100000;

%Intial conditions
x0 = [-8;7;27];

tspan = [t0 tf];

%State sapace model of a 2nd order linear differential equation
options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
[t,x] = ode45(@(t,x) LorentzSys63(t,x,param),tspan,x0');

size(t)
size(x)

%% Sample dynamical system
dsample = 1;
% dsample = sampleTime/dt;
% timeData = t(1:dsample:end);
% Nsamples = length(timeData);

%% True state variables
x1 = x(1:dsample:end,1);
x2 = x(1:dsample:end,2);
x3 = x(1:dsample:end,3);

xi = [x1,x2,x3]';

M = length(t);

%Compute histograms of the trajectory
nbins = 100;
figure;
%State X1
subplot(3,1,1)
histogram(x1,nbins)
tmeanx1 = mean(x1)

%State X1
subplot(3,1,2)
histogram(x2,nbins)
tmeanx2 = mean(x2)


%State X1
subplot(3,1,3)
histogram(x3,nbins)
tmeanx3 = mean(x3)

%%%%%%

%% Generate multi index array (graded lexicographic order)
deg = 3;

%Dimension of the basis
dim = nchoosek(n + deg,deg);
%Construct the multi index array

if n == 1
    alpha(:,1) = 0:1:deg;
else
    alpha = nd_multi_index_poly(n,deg);
end


%Generate multidimesnional basis matrix
X = zeros(M,dim);
for j = 1:dim
            alphajs = repmat(alpha(j,:),M,1);
            X(:,j) = prod(xi'.^(alphajs),2);
end

[Qmgs,Rmgs] = MGS_nD_samples(X);
CON_MGS = inv(Rmgs(1:dim,:))';%Coefficient matrix

%% TEST ORTHOGONALITY WITH A NEW DATA SET FROM THE SAME DYNAMICAL SYSTEM

%Intial conditions
x0test = [3;9;15];
tftest = 10;
dt = 0.0001;
tspan = t0:dt:tftest;

%State sapace model 
[x] = ode5(@(t,x) LorentzSys63(t,x,param),tspan,x0test');
t = tspan;
%% Sample dynamical system
m = length(t);
Mtest = 1e3;
dsample = floor(m/Mtest);

%% True state variables
x1test = x(1:dsample:end,1);
x2test = x(1:dsample:end,2);
x3test = x(1:dsample:end,3);
ttest = t(1:dsample:end);
xtest = [x1test,x2test,x3test]';

Ntest = length(ttest);

%Compute histograms of the test trajectory
nbins = 20;
figure;
%State X1
subplot(3,1,1)
histogram(x1test,nbins)
tmeanx1 = mean(x1test)

%State X1
subplot(3,1,2)
histogram(x2test,nbins)
tmeanx2 = mean(x2test)


%State X1
subplot(3,1,3)
histogram(x3test,nbins)
tmeanx3 = mean(x3test)


%% Check orthonormality
PolyMGS = zeros(dim,Mtest);
for d = 1:dim
    for s = 1:Mtest
        Poly_MGS = 0;
        
        for k=1:dim
            Poly_MGS = Poly_MGS + CON_MGS(d,k)*prod((xtest(:,s))'.^(alpha(k,:)),2);
        end
        
        PolyMGS(d,s) = Poly_MGS;
        
    end
end

G_MGS = 1/Mtest*PolyMGS*PolyMGS';

%Check orthogonality (Mutual coherence)
A = zeros(dim);
for c = 1:dim
    A(:,c) = G_MGS(:,c)/norm(G_MGS(:,c));
end
MC = max(max(abs(A - diag(diag(A)))))
%Check orthonormality
e_OrthN = norm(G_MGS - eye(dim),'f')




