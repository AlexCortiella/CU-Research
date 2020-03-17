function [z,C] = gramschmidtpolyMOD2(x,N,M)

for i=1:N
s(i,:)=x(i,:);
end

%This is the first orthonormal constant polynomial
e(1)=s(1,:)*conj(s(1,:).');
phi(1,:)=s(1,:)/sqrt(e(1));
C = zeros(N-1);

%Loop to generate next orthonormal polynomials
for i=2:N
    %th is the accumulation sum_{k=1}^{i-1}c_ir Phi_r, where c_ir are the
    %projection coefficients
    th(i,:)=zeros(1,M);
    for r=1:1:i-1
        C(i-1,r) = s(i,:)*conj(phi(r,:).');
        th(i,:) = th(i,:) + C(i-1,r)*phi(r,:);
    end
    th(i,:)=s(i,:)-th(i,:); %Compute the unnormalized basis
    e(i)=th(i,:)*conj(th(i,:).');
    phi(i,:)=th(i,:)/sqrt(e(i));%Normalize the basis
end

z=phi(1:N,:);

end

