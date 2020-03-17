function y = nlogspace(d1, d2, n)
%LOGSPACE Logarithmically spaced vector.
%   LOGSPACE(X1, X2) generates a row vector of 50 logarithmically
%   equally spaced points between decades exp(X1) and exp(X2).  If X2
%   is pi, then the points are between exp(X1) and pi.
%
%   LOGSPACE(X1, X2, N) generates N points.
%   For N = 1, LOGSPACE returns exp(X2).
%
%   Class support for inputs X1,X2:
%      float: double, single
%
%   See also LINSPACE, COLON.

%   Copyright 1984-2012 The MathWorks, Inc. 

if nargin == 2
    n = 50;
end

if d2 == pi || d2 == single(pi) 
    d2 = log(d2);
end

y = exp(linspace(d1, d2, n));
