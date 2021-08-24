function [sol, residual, reg_residual] = solveWBPDN(X, y, w, lambda, p_basis)

% Normalize columns of X
c_norm = vecnorm(X);
Wn = diag(1 ./ c_norm); %column-normalization matrix

if isempty(w)
    Ww = eye(p_basis);
else
    if length(w) ~= p_basis
        warning("The weight vector provided do not match the dimensions of X.Please, provide a weight vector so that len(w) = # columns of X");
                        
    else
         Ww = diag(1 ./ w);
    end
end

Wnw = Wn*Ww;
Xnw = X*Wnw;

sol_tilde = SolveBP(Xnw, y , p_basis, 100000, lambda , 1e-11);

%Unweight solution
sol = Wnw*sol_tilde;

%Compute residual
residual = norm(X*sol - y);
reg_residual = norm(sol_tilde, 1);

end

