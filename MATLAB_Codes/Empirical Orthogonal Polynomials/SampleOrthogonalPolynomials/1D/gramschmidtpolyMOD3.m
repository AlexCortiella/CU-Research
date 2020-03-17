function [z,C,Cn] = gramschmidtpolyMOD3(x,N,M)

for i=1:N
p(i,:)=x(i,:);
end
xdata = x(2,:);
%This is the first orthonormal constant polynomial
e(1)=p(1,:)*conj(p(1,:).');
% phi(1,:)=p(1,:)/sqrt(e(1));
phi(1,:)=p(1,:);

C = diag(ones(N,1));
Cn = C; Cn(1,1) = 1/sqrt(e(1));

%Loop to generate next orthonormal polynomials
for i=2:N
    %th is the accumulation sum_{k=1}^{i-1}c_ir Phi_r, where c_ir are the
    %projection coefficients
    th(i,:)=zeros(1,M);
    for r=1:1:i-1
        nint = p(i,:).*phi(r,:);
        dint = phi(r,:).*phi(r,:);
        plot(xdata,phi(r,:))
        nF = griddedInterpolant(xdata,nint);
        dF = griddedInterpolant(xdata,dint);
        nfun = @(t) nF(t);
        dfun = @(t) dF(t);
        [i,r]
        integral(nfun, -1, 1)
        integral(dfun, -1, 1)
        pause
        close
        C(i,r) = integral(nfun, -1, 1)/integral(dfun, -1, 1);
%         C(i,r) = p(i,:)*conj(phi(r,:).');
        th(i,:) = th(i,:) + C(i,r)*phi(r,:);
    end
    th(i,:)=p(i,:)-th(i,:); %Compute the unnormalized basis
    e(i)=th(i,:)*conj(th(i,:).');%Inner product of the unnormalized basis
    phi(i,:)=th(i,:);
%     phi(i,:)=th(i,:)/sqrt(e(i));%Normalize the basis
    Cn(i,:) = C(i,:)/sqrt(e(i));
end

z=phi(1:N,:);

end

