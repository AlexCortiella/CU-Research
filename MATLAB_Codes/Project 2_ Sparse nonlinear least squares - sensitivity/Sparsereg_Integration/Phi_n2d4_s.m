function [phi] = Phi_n2d4_s(x,index_pc)

%%% Build basis matrix %%%
phi = piset_monomial(x,index_pc);

end

