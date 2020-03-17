function [f df Hf] = myLogistic(x,A,y)
    ns = size(A,2);
    u = A'*x;
    f = sum(log(1+exp(-abs(u))) + double(u>0).*u - y.*u)/ns;
    if nargout >= 2
        df = A*(1./(1+exp(-u)) - y)/ns;
    end
    if nargout > 2
        Hf = bsxfun(@times,A,cosh(u'/2).^-2)*A'/(4*ns);
    end