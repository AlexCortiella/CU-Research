function [Qnormal,Rnormal] = MGS_nD_samples(x,deg,wfun,support)

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
        samp_inner = 1/M*(v'*v);
        R(i,i) = sqrt(samp_inner);
        q = v/sqrt(samp_inner);
        Q(:,i) = q;
        
        for j = i+1:N
            v = p(:,j);
            samp_inner = 1/M*(q'*v);
            R(i,j) = samp_inner;
            p(:,j) = p(:,j) - R(i,j)*q;
            
        end
end


Qnormal = Q;
Rnormal = R;

end

