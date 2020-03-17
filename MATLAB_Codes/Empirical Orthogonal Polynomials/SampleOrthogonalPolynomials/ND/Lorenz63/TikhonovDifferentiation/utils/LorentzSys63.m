function dxdt = LorentzSys63(t,x,param)

%Extract parameters
sigma = param(1);
rho = param(2);
beta = param(3);

%Transformation into a first order system
dxdt = [sigma*(x(2) - x(1)); x(1)*(rho - x(3)) - x(2); x(1)*x(2) - beta*x(3)];

end