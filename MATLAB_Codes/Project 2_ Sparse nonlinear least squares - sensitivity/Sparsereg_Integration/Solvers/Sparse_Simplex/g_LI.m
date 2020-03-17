function out=g_LI(A,b,x,S)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function finds the index of the variable (among the indices set S) 
% that causes the greatest decrease in the objective function
% value 
% ||Ax-b||^2
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
% b ............ the vector b associated with the objective fucntion
% x ............ current point
% S ............ indices set (from which an index should be chosen)
%
% OUTPUT
% ====================================================
% out ........... a 3-dimensional vector containing out(1) - the new value
%                 of the chosen variable, out(2) - the new objective
%                 function value and out(3) - the index of the chosen
%                 variable.


g=A'*(A*x-b);
norm_square_vector =(sum(A.^2))';

g_S=g(S);
norm_square_vector_S=norm_square_vector(S);

val_all=(g_S.^2)./norm_square_vector_S;
[stam,ind]=max(val_all);

val=x(S(ind))-g(S(ind))/norm_square_vector(S(ind));

fun_val=norm(A*x-b-g(S(ind))/norm_square_vector(S(ind))*A(:,S(ind)))^2;
out(1)=val;
out(2)=fun_val;
out(3)=S(ind);