function [dxS] = IntegrateModel_s(t,xs,Xi,index_pc)


p = size(index_pc,1);
n = size(index_pc,2);
x = xs(1:n);
s = xs(n+1:end);
%Sensitivity and Jacobian matrices
S = reshape(s,n,p);%dx_i/dXi_j (Nstates x Nbasis)
J = Jacobian_n2d4(x); %dPhi_j/dx_i (Nbasis x Nstates)
%Basis matrix
PhiM = Phi_n2d4_s(x',index_pc);%[Phi, Phi, Phi,...,Phi] (Nbasis x Nstates)
%Compute time derivative of sensitivity
dS = J*S*Xi + PhiM;%(Nbasis x Nstates)
ds = dS(:);%(Nbasis*Nstates x 1)

phi = (piset_monomial(x',index_pc))';%(1 x Nbasis)
dx = phi*Xi;%(Nstates x 1);
size(dx)
dxS = [dx';ds];
end

