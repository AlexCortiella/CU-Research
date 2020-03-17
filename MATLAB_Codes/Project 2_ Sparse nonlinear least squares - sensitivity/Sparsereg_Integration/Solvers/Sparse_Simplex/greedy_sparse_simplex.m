function [X,fun_val]=greedy_sparse_simplex(f,g,s,N,x0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function employes the greedy sparse simplex method on the problem
% (P) min \{ f(x): ||x||_0 <=s\}
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
% f ............ the objective function (which is a function of the
%                decision variable vector x)
% g ............ a function performing one-dimensional optimization of 
%                the function f. Its input consists of the pair (x,S) where
%                x is the input decision variables vector and S is a set of
%                indices on which the optimization is performed
% s ............ sparsity level
% N ............ maximum number of iterations
% x0 ........... initial vector
%
% OUTPUT
% ====================================================
% X ............ The sequence of iterates generated by the method
% fun_val ...... The obtained function value (of the last iterate)



n=length(x0);
x=x0;

X=[];
fold=Inf;
fun_val=-Inf;
iter_stuck=0;
for iter=1:N
    if (abs(fold-fun_val)<1e-8)
        iter_stuck=iter_stuck+1;
    end
    if(iter_stuck==5)
        break
    end


    fold=fun_val;

    d=nnz(x);


    if (d>s)
        ok=1;
        x(s+1:n)=0;
    end
    X=[X,x];
    if(d<s)
        ok=0;
        min_funval=Inf;
        out=g(x,1:n);
        val=out(1);
        fun_val=out(2);
        ind=out(3);
        if(fun_val<min_funval)
            min_index=ind;
            min_funval=fun_val;
            min_val=val;
        end
        x(min_index)=min_val;
    end
    if (d==s)
        I1=find(x);
        min_funval=Inf;
        for i=1:s
            xtilde=x;
            xtilde(I1(i))=0;
            out=g(xtilde,1:n);
            val=out(1);
            fun_val=out(2);
            ind=out(3);
            if(fun_val<min_funval)
                min_index_out=I1(i);
                min_index_in=ind;
                min_funval=fun_val;
                min_val=val;
            end
           
        end
        ok=0;
        if(x(min_index_in)==0)
            ok=1;
        end
        x(min_index_out)=0;

        x(min_index_in)=min_val;
    end
    fun_val=f(x);
    fprintf('iter=%3d  fun_val = %5.5f   change = %d\n',iter,fun_val,ok);
end
fun_val=f(x);
