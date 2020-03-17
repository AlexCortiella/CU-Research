function out=g_QU(A,c,x,Q)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function finds the index of the variable (among the indices set Q) 
% that causes the greatest decrease in the objective function
% \sum_{i=1}^m ((A(i,:)*x)^2-c)^2
% The output is the three-dimensional vector out comprising the new value of the index,
% the new objective function value, and the index causing the greatest
% decrease
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
% A ............ the matrix A associated with the objective function
% c ............ the vector c associated with the objective fucntion
% x ............ current point
% S ............ indices set (from which an index should be chosen)
%
% OUTPUT
% ====================================================
% out ........... a 3-dimensional vector containing out(1) - the new value
%                 of the chosen variable, out(2) - the new objective
%                 function value and out(3) - the index of the chosen
%                 variable.

    
    s=size(A);
    n=s(2);
    A2=A.*A;
    A3=A.*A2;
    A4=A.*A3;
    w1=sum(A4)';
    w2=A2'*c;
    u1=A*x;
    u2=u1.^2;
    u4=u2.^2;
    v5=w1;
    v4=4*A3'*u1;
    v3=6*A2'*u2-2*w2;
    v2=4*A'*((u2-c).*u1);
    v1=sum((u2-c).^2)*ones(n,1);

    V=[v1,v2,v3,v4,v5];
    [out2,fun]=solve_minimum_quartic(V);

    fun_limit=fun(Q);
    [fval,ind]=min(fun(Q));
    
    out=zeros(3,1);
    out(1)=x(Q(ind))+out2(Q(ind));
    out(2)=fval;
    out(3)=Q(ind);
    

