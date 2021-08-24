function [y,x] = bspline_basis(j,n,t,x)
% B-spline basis function value B(j,n) at x.
%
% Input arguments:
% j:
%    interval index, 0 =< j < numel(t)-n
% n:
%    B-spline order (2 for linear, 3 for quadratic, etc.)
% t:
%    knot vector
% x (optional):
%    value where the basis function is to be evaluated
%
% Output arguments:
% y:
%    B-spline basis function value, nonzero for a knot span of n

y = bspline_basis_recurrence(j,n,t,x);

function y = bspline_basis_recurrence(j,n,t,x)

    y = zeros(size(x));
    if n > 1
        b = bspline_basis(j,n-1,t,x);
        dn = x - t(j+1);
        dd = t(j+n) - t(j+1);
        if dd ~= 0  % indeterminate forms 0/0 are deemed to be zero
            y = y + b.*(dn./dd); %add first term
        end
        b = bspline_basis(j+1,n-1,t,x);
        dn = t(j+n+1) - x;
        dd = t(j+n+1) - t(j+1+1);
        if dd ~= 0
            y = y + b.*(dn./dd); %add second term
        end
    elseif t(j+2) < t(end)  % treat last element of knot vector as a special case
        y(t(j+1) <= x & x < t(j+2)) = 1;
    else
        y(t(j+1) <= x) = 1;
        
    end
    
