function [dx] = IntegrateModelDifferror(t,x,Xi,index_pc)

phi = piset_monomial(x',index_pc);

dx = Xi*phi;
end

