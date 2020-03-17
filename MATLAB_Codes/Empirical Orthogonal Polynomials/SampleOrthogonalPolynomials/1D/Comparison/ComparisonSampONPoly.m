%% Comparison for computing orthonormal polynomials

%% Test
clc
close all
clear all

% addpath('./aPC_Matlab_Toolbox')
Degree = 3;
Nsamp = 1e6;
a = -1;
b = 1;
% xi = a + (b-a)*rand(Nsamp,1);
% xi = randn(Nsamp,1);

pdf = 'uniform';
% pdf = 'normal';
% mu = 0;
% sigma = 1;

xi = randraw(pdf, [a, b], Nsamp);
% xi = randraw(pdf, [mu, sigma], Nsamp);

xi = xi';

%% Generate sample-Vandermonde matrix
X = zeros(Nsamp,Degree+1);

for d = 0:Degree
    X(:,d+1) = xi.^(d);
end

%% MGS algorithm to compute coefficients (non-normalized)
[Qmgs,Rmgs] = MGS_1D_samples(xi,Degree);
CON_MGS = inv(Rmgs(1:Degree+1,:))';%Coefficient matrix

%% QR algorithm to compute coefficients (non-normalized)
[R] = qr(X,0);
C = inv(R(1:Degree+1,:))';%Coefficient matrix

%Normalize coefficients
CON_QR = zeros(Degree+1,Degree+1);
for d = 0:Degree
P_norm=0;
    for i=1:Nsamp      
        Poly=0;
        for k=0:d
            Poly=Poly+C(d+1,k+1)*xi(i)^k;     
        end
        P_norm=P_norm+Poly^2/Nsamp;        
    end
    P_norm=sqrt(P_norm);
    for k=0:d
        CON_QR(d+1,k+1)=C(d+1,k+1)/P_norm;
    end
end

%% Moment matrix (iterative) to compute coefficients
[CON_Mom, CMon_Mom] = aPC_samp(xi, Degree);

%% Moment matrix (Golub-Welsch) to compute coefficients
% Raw Moments of the Input Data
for i=0:(2*Degree+1)
    m(i+1)=sum(xi.^i)/Nsamp; 
end
% Polynomial up to degree Degree
    % Generate moment matrix
    %Definition of Moments Matrix Mm (Bulid the moment matrix)
    
Mm = zeros(Degree+1,Degree+1);
for i = 0:Degree
    for j = 0:i
        Mm(i+1,j+1) = 1/Nsamp*sum(xi.^(i).*xi.^(j));
    end
end
%Symmetrize matrix
Mm = (Mm + Mm' - diag(diag(Mm)));
[UChol] = chol(Mm);

% for j = 1:Degree
%     if j == 1
%         abChol(j,1) = -UChol(j,j+1)/UChol(j+1,j+1);
%         abChol(j,2) = UChol(j,j)/UChol(j+1,j+1);
%     else
%         abChol(j,1) = UChol(j-1,j)/UChol(j,j) - UChol(j,j+1)/UChol(j+1,j+1);
%         abChol(j,2) = UChol(j,j)/UChol(j+1,j+1);
%     end
% end

CON_Chol = inv(UChol)';

%% Stieltjes pocedure
ab = stieltjesSamplesC(xi',Degree+2)
abStieltjes_Monic = ab(1:end-1,:);%Monic
abStieltjes_ON = abStieltjes_Monic;
for i=1:Degree+1
    abStieltjes_ON(i,2) = sqrt(ab(i+1,2));
end
    

%Monic to orthonormal
% abStieltjes_ON = [abStieltjes_Monic(:,1),sqrt(abStieltjes_Monic(:,2))];

%Only for comparison to Legendre
alpha = 0;
beta = 0;
% ab_jacobi_Monic = r_jacobi(Degree+1,alpha,beta);
% ab_jacobi_ON = [ab_jacobi_Monic(:,1),sqrt(ab_jacobi_Monic(:,2))];

%% Generate sampled polynomial
N = 1e6; 
x = a + (b-a)*rand(N,1);
x = randn(N,1);
% pdf = 'normal';
% mu = 0;
% sigma = 1;
x = randraw(pdf, [a,b], N);
% x = randraw(pdf, [mu,sigma], N);



%Given full coefficients
for d = 0:Degree
    for s = 1:N
        Poly_QR = 0;
        Poly_Mom = 0;
        Poly_Chol = 0;
        Poly_MGS = 0;
        
        for k=0:d
            Poly_QR = Poly_QR + CON_QR(d+1,k+1)*x(s)^k;     
            Poly_Mom = Poly_Mom + CON_Mom(d+1,k+1)*x(s)^k;
            Poly_Chol = Poly_Chol + CON_Chol(d+1,k+1)*x(s)^k;
            Poly_MGS = Poly_MGS + CON_MGS(d+1,k+1)*x(s)^k;
        end
        
        PolyQR(d+1,s) = Poly_QR;
        PolyMom(d+1,s) = Poly_Mom;
        PolyChol(d+1,s) = Poly_Chol;
        PolyMGS(d+1,s) = Poly_MGS;
        
    end
end

%Given three term recursion coefficients
PolyStieltjes = TTRec_on(x',abStieltjes_ON);


G_QR = 1/N*PolyQR*PolyQR';
eG_QR = norm(G_QR - eye(Degree+1),'f')

G_Mom = 1/N*PolyMom*PolyMom';
eG_Mom = norm(G_Mom - eye(Degree+1),'f')

G_Chol = 1/N*PolyChol*PolyChol';
eG_Chol = norm(G_Chol - eye(Degree+1),'f')

G_Stieltjes = 1/N*PolyStieltjes*PolyStieltjes';
eG_Stieltjes = norm(G_Stieltjes - eye(Degree+1),'f')

G_MGS = 1/N*PolyMGS*PolyMGS';
eG_MGS = norm(G_MGS - eye(Degree+1),'f')