function dxdt = duffing(t,x,param)

%Setup parameters
gamma = param(1); kappa = param(2); epsilon = param(3);

%Transformation into a first order system
dxdt = [x(2); -gamma * x(2) - kappa * x(1) - epsilon * x(1)^3];

end