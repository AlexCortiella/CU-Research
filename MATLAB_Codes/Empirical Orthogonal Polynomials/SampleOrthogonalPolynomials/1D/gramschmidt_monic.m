function [Qmonic,Rmonic] = gramschmidt_monic(x,deg,wfun,support)

%Number of data points
M = length(x);
%Number of monomials
N = deg+1;

%Evaluate the weight function at the sample points
w = feval(wfun,x);

%Extract support limits
a = support(1);
b = support(2);

%Polynomial basis
X = zeros(N,M);
for i = 1:N
    X(i,:) = x.^(i-1);
end

%Generate matrices
p = X;
phi(1,:)=p(1,:);
C = diag(ones(N,1));
th = zeros(N,M);

%Loop to generate next orthonormal polynomials
for i=2:N
    %th is the accumulation sum_{k=1}^{i-1}c_ir Phi_r, where c_ir are the
    %projection coefficients
    for r=1:1:i-1
        %Compute numerator and denominator integrals using interpolation
        nint = p(i,:).*phi(r,:).*w;
        dint = phi(r,:).*phi(r,:).*w;
        nF = griddedInterpolant(x,nint);
        dF = griddedInterpolant(x,dint);
        nfun = @(t) nF(t);
        dfun = @(t) dF(t);
        %Compute projection coefficients
        C(i,r) = integral(nfun, a, b)/integral(dfun, a, b);
        %Accumulate terms
        th(i,:) = th(i,:) + C(i,r)*phi(r,:);
    end
    th(i,:)=p(i,:)-th(i,:); %Compute the monic basis
    phi(i,:)=th(i,:);
end

z=phi(1:N,:);

Qmonic = z';
Rmonic = C';

end

