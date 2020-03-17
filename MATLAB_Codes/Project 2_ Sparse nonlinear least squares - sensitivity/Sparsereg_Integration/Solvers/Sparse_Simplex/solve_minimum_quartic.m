function [out,fun]=solve_minimum_quartic(a)
%this function minimizes  quartic  funcctions of the form
%a(i,1)+a(i,2)*x+a(i,3)*x^2+a(i,4)*x^3+a(i,5)*x^4

s=size(a);
m=s(1);
n=s(2);

out=solve_cubic([a(:,2),2*a(:,3),3*a(:,4),4*a(:,5)]);
ro=real(out);
v=kron(a(:,1),ones(1,3))+kron(a(:,2),ones(1,3)).*ro+kron(a(:,3),ones(1,3)).*ro.^2+kron(a(:,4),ones(1,3)).*ro.^3+kron(a(:,5),ones(1,3)).*ro.^4;
[fun,ind]=min(v');
fun=fun';
S=sparse([1:m]',ind,ones(m,1),m,3);
out=full(sum((S.*ro)')');
% fun=full(sum((S.*)')');
% out=ro(ind);  

