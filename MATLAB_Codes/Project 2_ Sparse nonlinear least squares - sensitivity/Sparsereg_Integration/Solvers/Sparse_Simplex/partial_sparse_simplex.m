function [X,fun_val]=partial_sparse_simplex(f,f_grad,g,s,N,x0)

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
        P=randperm(n);
        x(P(1:n-s))=0;
%         error('error - too many nonzero elements');
        ok=0;
    end
    X=[X,x];
    if(d<s)
        min_funval=Inf;
        out=g(x,1:n);
        min_val=out(1);
            min_funval=out(2);
            min_index=out(3);
        x(min_index)=min_val;
        ok=0;
    end
    if (d==s)
        ok=0;
        I1=find(x);
        I0=setdiff(1:n,I1);
      
        out=g(x,I1);
            min_val=out(1);
            min_funval=out(2);
            min_index=out(3);

        [stam,ind] = min(abs(x(I1)));
        i_index = I1(ind);
        gradient = f_grad(x);
        [stam,ind]=max(abs(gradient(I0)));
        j_index = I0(ind);
        xtilde=x;
        xtilde(i_index)=0;
        out=g(xtilde,j_index);
        val=out(1);
        fun_val=out(2);
        if (fun_val<min_funval+1e-8)
            ok=1;
            x(i_index)=0;
            x(j_index)=val;
        else
            x(min_index)=min_val;
        end
       
    end
    fun_val=f(x);

    fprintf('iter=%3d  fun_val = %5.5f change = %d\n',iter,fun_val,ok);
end
fun_val=f(x);
