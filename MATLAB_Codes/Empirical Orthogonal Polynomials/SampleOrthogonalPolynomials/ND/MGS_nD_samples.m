function [Qnormal,Rnormal] = MGS_nD_samples(X)

%Number of samples
M = size(X,1);
%Number of basis
N = size(X,2);
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

