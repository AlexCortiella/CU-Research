%% Empirical orthogonal polynomials in several variables (method of moments)
clc;
close all;
clear all;
rng(50);
%% Setup parameters
%Number of variables
Nv = 2;
% Degree of the polynomial
deg = 5;

%% Generate samples in Nv variables
%Number of samples
N = 1e6;

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

%% Generate multivariate moments from the multi index array

%Number of multivariate moments to generate the moment matrix

% Exportable method for other software (not efficient in matlab)
% m = zeros(dim,1);
% for j = 1:dim
%     temp2 = 0;
%     for s = 1:N
%         temp = 1;
%         for i = 1:Nv
%             temp = temp*(xi(i,s)')^(alpha(j,i));
%         end
%         temp2 = temp2 + 1/N*temp;
%     end
%     m(j) = temp2;
% end

%Efficient method to compute multivariate empirical moments in matlab
m = zeros(dim,1);
for j = 1:dim
            alphajs = repmat(alpha(j,:),N,1);
            m(j) = 1/N*sum(prod(xi'.^(alphajs),2));
end

% Mm2 = zeros(dim,dim);
% for i = 1:dim
%     for j = 1:dim
%         Mm2(i,j) = 1/N*sum(xi.^(i-1).*xi.^(j-1));
%     end
% end

Mm = zeros(dim,dim);
phis = zeros(N,dim);
%Compute lower triangular part only
for i = 1:dim
    alphajs = repmat(alpha(i,:),N,1);
    phis(:,i) = prod(xi'.^(alphajs),2);
    for j = 1:i
        Mm(i,j) = 1/N*sum(phis(:,i).*phis(:,j));
    end
end
%Symmetrize matrix
Mm = (Mm + Mm' - diag(diag(Mm)));

%% Compute the coefficients of the multi-dimensional orthogonal polynomial basis
% %Cholesky factorization of the moment matrix
[R] = chol(Mm);
CO_mom = inv(R)';
resI = norm(CO_mom*R - eye(dim))
%Normalize coefficients
CON_mom = zeros(dim,dim);
for d = 1:dim
P_norm=0;
    for i=1:N      
        Poly=0;
        for k=1:d
            Poly=Poly+CO_mom(d,k)*prod((xi(:,i))'.^(alpha(k,:)),2);
        end
        P_norm=P_norm+Poly^2/N;        
    end
    P_norm=sqrt(P_norm);
    for k=1:d
        CON_mom(d,k)=CO_mom(d,k)/P_norm;
    end
end

%% Test orthonormality
Ntest = 1e5;
xitest = mvnrnd(MU,SIGMA,Ntest)';

%% Check orthonormality
PolyMom = zeros(dim,Ntest);
for d = 1:dim
    for s = 1:Ntest
        Poly_Mom = 0;
        
        for k=1:dim
            Poly_Mom = Poly_Mom + CO_mom(d,k)*prod((xitest(:,s))'.^(alpha(k,:)),2);
        end
        
        PolyMom(d,s) = Poly_Mom;
        
    end
end

G_Mom = 1/Ntest*PolyMom*PolyMom';

%Check orthogonality (Mutual coherence)
A = zeros(dim);
for c = 1:dim
    A(:,c) = G_Mom(:,c)/norm(G_Mom(:,c));
end
MC = max(max(abs(A - diag(diag(A)))))
%Check orthonormality
e_OrthN = norm(G_Mom - eye(dim),'f')







