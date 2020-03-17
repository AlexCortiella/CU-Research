function h = myLogistic_Hv(v,x,A)
    ns = size(A,2);
    u = A'*[x v];
    h = A*(cosh(u(:,1)/2).^-2.*u(:,2))/(4*ns);    