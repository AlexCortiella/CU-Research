%% Empirical orthogonal polynomials in several variables (method of moments)
clc;
close all;
clear all;
rng(50);
%% Setup parameters
%Number of variables
Nv = 2;
% Degree of the polynomial
deg = 3;

%% Generate samples in Nv variables
%Number of samples
N = 1e5;

%% UNIFORM DISTRIBUTION [-1,1]
%Define suport for each variable
% a = -ones(Nv,1);
% b = ones(Nv,1);

%Generate samples given an arbitrary distribution (correlated or uncorrelated)
% xi = a + (b-a).*rand(Nv,N);%This is the un-correlated case

%% NORMAL DISTRIBUTION [MU,SIGMA]
MU = [2 3];
s11 = 1;
s22 = 3;
s12 = 1.5;%correlation term
SIGMA = [s11 s12; s12 s22];
rng('default')  % For reproducibility
xi = mvnrnd(MU,SIGMA,N)';
nbins = 50;
hist3(xi',[nbins,nbins],'CDataMode','auto','FaceColor','interp')
%% Generate multi index array (graded lexicographic order)
%Dimension of the basis
dim = nchoosek(Nv + deg,deg);
%Construct the multi index array

if Nv == 1
    alpha(:,1) = 0:1:deg;
else
    alpha = nd_multi_index_poly(Nv,deg);
end

size(xi)

%Generate multidimesnional basis matrix
X = zeros(N,dim);
for j = 1:dim
            alphajs = repmat(alpha(j,:),N,1);
            X(:,j) = prod(xi'.^(alphajs),2);
end

[Qmgs,Rmgs] = MGS_nD_samples(X);
CON_MGS = inv(Rmgs(1:dim,:))';%Coefficient matrix

%% Test orthonormality
Ntest = 1e5;
xitest = mvnrnd(MU,SIGMA,Ntest)';
%% Check orthonormality
PolyMGS = zeros(dim,Ntest);
for d = 1:dim
    for s = 1:Ntest
        Poly_MGS = 0;
        
        for k=1:dim
            Poly_MGS = Poly_MGS + CON_MGS(d,k)*prod((xitest(:,s))'.^(alpha(k,:)),2);
        end
        
        PolyMGS(d,s) = Poly_MGS;
        
    end
end

G_MGS = 1/Ntest*PolyMGS*PolyMGS';

%Check orthogonality (Mutual coherence)
A = zeros(dim);
for c = 1:dim
    A(:,c) = G_MGS(:,c)/norm(G_MGS(:,c));
end
MC = max(max(abs(A - diag(diag(A)))))
%Check orthonormality
e_OrthN = norm(G_MGS - eye(dim),'f')







