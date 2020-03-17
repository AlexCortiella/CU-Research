function [Phi] = Phi_n2d4(x,index_pc)

n = length(x);

%%% Build basis matrix %%%
phi = piset_monomial(x,index_pc);
Phi = repmat(phi,1,n);

end

