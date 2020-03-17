function [dx] = integrateApprxDynamics(t,x,Xi,index_pc,x0)

phi = piset_monomial(x0',index_pc);

dx = phi'*Xi;

end

