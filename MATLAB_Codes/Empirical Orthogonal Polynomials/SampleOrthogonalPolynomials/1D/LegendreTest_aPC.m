%% ORTHOGONAL POLYNOMIALS

clc;
close all;
clear all;
addpath('./OPQ_utility','./OPQ_tablesandfigures','./OPQ_quadrature','./OPQ_orthpol','./OPQ_examplesandtests','./aPC_Matlab_Toolbox')

Degree = 3;
Nsamp = 1e6;

%% Generate samples with specific density

%Define support
a = -1;
b = 1;
xi = a + (b-a)*rand(Nsamp,1);
[Coeff_ONB,Coeff_MonicB] = aPC_samp(xi, Degree);

%% Plot orthogonal polynomials
N = 1e6;
dN = 1000;
x = a + (b-a)*rand(N,1);
Poly_Norm = zeros(Degree+1,N);
% Poly_Monic = zeros(Degree+1,N);

for d = 0:Degree
    for s = 1:N
        PolyN = 0;
        PolyM = 0;
        
        for k=0:d
            PolyN = PolyN + Coeff_ONB(d+1,k+1)*x(s)^k;     
%             PolyM = PolyM + Coeff_MonicB(d+1,k+1)*x(s)^k;     
        end
        
        Poly_Norm(d+1,s) = PolyN;
%         Poly_Monic(d+1,s) = PolyM;
        
    end
end

xs = x(1:dN:end);
[xs,indx] = sort(x);
figure
for d = 0:Degree
    subplot(1,2,1)
    plot(xs,Poly_Norm(d+1,indx))
    hold on
end
% for d = 0:Degree
%     subplot(1,2,2)
%     plot(x(1:dN:end),Poly_Monic(d+1,1:dN:end),'Marker','.')
%     hold on
% end

%Generate Gram matrix and take average over samples
GNorm = 1/N*Poly_Norm*Poly_Norm';
% GMonic = 1/N*Poly_Monic*Poly_Monic';

% for col = 1:size(GMonic,2)
%     GMonic(:,col) = GMonic(:,col)/norm(GMonic(:,col));
% end

eGNorm = norm(GNorm - eye(Degree+1),'f')
% eGMonic = norm(GMonic - eye(Degree+1),'f')




