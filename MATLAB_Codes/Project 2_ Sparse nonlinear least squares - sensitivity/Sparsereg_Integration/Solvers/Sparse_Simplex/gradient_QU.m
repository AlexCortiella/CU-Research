function out=gradient_QI(A,c,x)

out=4*A'*((A*x).*((A*x).^2-c));