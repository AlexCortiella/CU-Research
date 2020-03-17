function dxdt = DuffingODE(t,x,param)

%Setup parameters
omega = param(1); epsilon = param(2); xi = param(3);

%Transformation into a first order system
dxdt = [x(2); -2*omega*xi*x(2) - omega^2*(x(1) + epsilon*x(1).^3)];

end