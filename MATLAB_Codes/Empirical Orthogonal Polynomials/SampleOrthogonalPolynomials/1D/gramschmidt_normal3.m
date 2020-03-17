function [Qnormal,Rnormal] = gramschmidt_normal3(x,deg,wfun,support)

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
X = zeros(M,N);
for i = 1:N
    X(:,i) = x.^(i-1)';
end

%Generate matrices
p = X;
R = zeros(N);
Q = zeros(M,N);

%Loop to generate next orthonormal polynomials
for j=1:N
    v = p(:,j);
    %th is the accumulation sum_{k=1}^{i-1}c_ir Phi_r, where c_ir are the
    %projection coefficients
    for i=1:1:j-1
        q = Q(:,i);
        %Compute numerator and denominator integrals using interpolation
        int = q.*v.*w;
        int2 = q.*q.*w;
        F = griddedInterpolant(x,int);
        fun = @(t) F(t);
        F2 = griddedInterpolant(x,int2);
        fun2 = @(t) F2(t);
        %Compute projection coefficients
        R(i,j) = integral(fun, a, b)/integral(fun2, a, b);
        %Accumulate terms
        v = v - R(i,j)*q;
    end
    int = v.*v.*w;
    F = griddedInterpolant(x,int);
    fun = @(t) F(t);
    norm2 = integral(fun, a, b);
    Q(:,j) = v/sqrt(norm2);
    R(j,j) = sqrt(norm2);
end

Qnormal = Q;
Rnormal = R;

end

