%% ORTHOGONAL POLYNOMIALS

clc;
close all;
clear all;
addpath('./OPQ_utility','./OPQ_tablesandfigures','./OPQ_quadrature','./OPQ_orthpol','./OPQ_examplesandtests','./aPC_Matlab_Toolbox')

Degree = 5;
Nsamp = 1e6;

%% Generate samples with specific density

%Define support
a = -1;
b = 1;
xi = a + (b-a)*rand(Nsamp,1);
N = length(xi);
C = zeros(Degree+1,Degree+1);
C(1,1) = 1;

for d = 0:Degree
    Ctemp = 0;
    for s = 1:N
        Ctemp = Ctemp + xi(s)^(d+1)/N;
    end
    C(d+1,1) = Ctemp;
end
    
    
    
% for d = 0:Degree
%     Ctemp=0;
%     for i=1:Nsamp        
%         Poly=0;
%         for k=0:Degree
%             Poly = Poly + C(degree+1,k+1)*xi(i)^k;     
%         end
%         Ctemp = Ctemp + Poly*xi(i)^k/Nsamp;%Sum over all sample points        
%     end
% end

figure
for d = 0:Degree
    subplot(1,2,1)
    plot(x(1:dN:end),Poly_Norm(d+1,1:dN:end),'Marker','.')
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




