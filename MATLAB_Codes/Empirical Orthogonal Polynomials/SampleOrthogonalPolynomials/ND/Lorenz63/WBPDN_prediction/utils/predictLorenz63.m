function [dx] = predictLorenz63(t,x,Xi,ployorder)

Ndofs = length(x);
index_pc = nD_polynomial_array(Ndofs,ployorder);

C = piset_monomial(x',index_pc);

dx = (C'*Xi)';

end

