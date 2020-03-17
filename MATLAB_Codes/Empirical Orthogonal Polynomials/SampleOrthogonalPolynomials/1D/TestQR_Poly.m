%% Test
clc
close all
clear all
% addpath('./aPC_Matlab_Toolbox')
Degree = 3;
Nsamp = 1e6;
a = -1;
b = 1;
xi = a + (b-a)*rand(Nsamp,1);
xi = xi';
X = zeros(Nsamp,Degree+1);

for d = 0:Degree
    X(:,d+1) = xi.^(d);
end

% [Q,R] = qr(X);
[R] = qr(X,0);


C = inv(R(1:Degree+1,:))';
CON = zeros(Degree+1,Degree+1);

[xis,indx] = sort(xi);
% for d = 0:N
%     plot(xis,Q(indx,d+1));
%     hold on
% end
% for d = 0:Degree
%     plot(xis,ON(indx,d+1));
%     hold on
% end

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
        CON(d+1,k+1)=C(d+1,k+1)/P_norm;
    end
end

N = 1e6;
x = a + (b-a)*rand(N,1);

for d = 0:Degree
    for s = 1:N
        PolyN = 0;
%         PolyM = 0;
        
        for k=0:d
            PolyN = PolyN + CON(d+1,k+1)*x(s)^k;     
%             PolyM = PolyM + Coeff_MonicB(d+1,k+1)*x(s)^k;     
        end
        
        Poly_Norm(d+1,s) = PolyN;
%         Poly_Monic(d+1,s) = PolyM;
        
    end
end

GNorm = 1/N*Poly_Norm*Poly_Norm'
eGNorm = norm(GNorm - eye(Degree+1),'f')


