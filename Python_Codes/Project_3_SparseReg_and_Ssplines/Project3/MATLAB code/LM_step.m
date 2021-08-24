function [x,reg_lm] = LM_step(x, reg_lm, loss_tr, Jac, res)


%Compute increment
Hess = Jac'*Jac;
grad = Jac'*res;
LHS = Hess + reg_lm*diag(diag(Jac));
RHS = -grad;
hlm = LHS\RHS;

%% Adaptive LM
loss_old = loss_tr(k);        
x_try = x + hlm;
%Compute new loss
loss_try = compute_loss(x_try);
%Compute adaptive rho parameter in LM (reference: The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems)
num = loss_old - loss_try;
den = hlm'*((reg_lm*diag(diag(Jac)))*hlm - rm);
%         den = hlm'*(reg_lm*hlm + r);
rho = num/den;

if rho > tol_update
    xi(:,k+1) = xi_try;
    reg_lm = max(reg_lm/Ldown,1e-7);
else
    xi(:,k+1) = xi(:,k);
    reg_lm = min(reg_lm*Lup,1e7);
    fprintf('Step rejected! rho = %0.6f \n\n',rho)
end

end

