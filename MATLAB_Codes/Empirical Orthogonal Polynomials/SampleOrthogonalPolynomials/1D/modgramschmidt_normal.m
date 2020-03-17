function [Qnormal,Rnormal] = modgramschmidt_normal(x,deg,wfun,support)

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
for i=1:N
        v = p(:,i);
        int = v.*v.*w;
        F = griddedInterpolant(x,int);
        fun = @(t) F(t);
        norm2 = integral(fun, a, b);
        R(i,i) = sqrt(norm2);
        q = v/sqrt(norm2);
        Q(:,i) = q;
        
        for j = i+1:N
            v = p(:,j);
            int = q.*v.*w;
            F = griddedInterpolant(x,int);
            fun = @(t) F(t);
            R(i,j) = integral(fun,a,b);
            p(:,j) = p(:,j) - R(i,j)*q;
            
        end
end


Qnormal = Q;
Rnormal = R;

end

