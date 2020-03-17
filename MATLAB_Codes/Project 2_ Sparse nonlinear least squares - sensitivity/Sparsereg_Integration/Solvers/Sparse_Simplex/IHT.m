function [x,fx]=IHT(f,g,s,L,x0,N);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function employs N iterations of the iterative hard-thresholding method for solving the sparsity-constrained 
% problem min{f(x):||x||_0 <=s}
%
% Based on the paper
% Amir Beck and Yonina Eldar, "Sparsity Constrained Nonlinear Optimization: Optimality Conditions and Algorithms"
% -----------------------------------------------------------------------
% Copyright (2012): Amir Beck and Yonina Eldar
% 
% Distributed under the terms of 
% the GNU General Public License 2.0.
% 
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
%-------------------------------------------------------------
% INPUT
%===================================================
% f ............ the objective function
% g ............ the gradient of the objective function
% s ............ the sparsity level 
% L ............ upper bound on the Lipschitz constant
% x0 ........... initial vector 
% N ............ number of iterations 
%
% OUTPUT
% ====================================================
% x ............. the output of the IHT method
% fval .......... objective function value of the obtained vector

n=length(x0);
x=x0;
for i=1:N
    x=x-1/L*g(x);
    [xsort,isort]=sort(abs(x));
    xnew=zeros(n,1);
    xnew(isort(n-s+1:n))=x(isort(n-s+1:n));
    x=xnew;
    fprintf('iter = %5d value=%5.10f\n',i,f(x));
end
fx=f(x);
