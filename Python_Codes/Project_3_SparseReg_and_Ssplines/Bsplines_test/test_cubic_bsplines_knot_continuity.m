%% Example of truncated cubic splines
clc;
close all;
clear all;

%% Generate data
dx = 0.01;
x = 0:dx:1;
y = sin(10*x);

m_samples = length(x);

plot(x,y,'ro')
pause
close;

%% Cubic splines setup
d = 3; %degree of the polynomial
M = d+1; %order of spline
n_knots = 5;

% knots = [0.25, 0.5, 0.75];%interior knots
% knots = linspace(0,1,n_knots);%interior knots
% knots = [0,0,0,0,1];%C^(M-m-1) = C^-1 continuity --> discontinuous at x = 0 (m is knot multiplicity)
% knots = [0,0,0,0.5,1];%C^(M-m-1) = C^0 continuity --> continuous at x = 0 
% knots = [0,0,1/3,2/3,1];%C^(M-m-1) = C^1 continuity --> continuous 1st derivative at x = 0 
knots = [0,0.25,0.5,0.75,1];%C^(M-m-1) = C^2 continuity --> continuous 2nd derivative at x = 0 


% % knots = [x(1), x(1),x(1),x(1),x,x(end),x(end),x(end),x(end)];
% knots = [x(1)-4*dx, x(1)-3*dx,x(1)-2*dx,x(1)-dx,x,x(end)+dx,x(end)+2*dx,x(end)+3*dx,x(end)+4*dx];

% knots = x;

K = length(knots);

n_basis = K+M;

%% Generate truncated power series for splines
[B,~] = bspline_basismatrix(M,knots,x);

%% Solve for coefficients
theta = B\y';

y_spline = B*theta;

%% Verify model

figure;
plot(x,y,'ro')
hold on
plot(x,y_spline,'b')
plot(knots,0,'Marker','s','MarkerFaceColor','k','Markersize',8);

%% Plot Bsplines recurrence

figure;
for l = 1:M
    subplot(M,1,l);
    [B,~] = bspline_basismatrix(l,knots,x);
    for c = 1:size(B,2)
        plot(x,B(:,c))
        hold on
        plot(knots,0,'Marker','s','MarkerFaceColor','k','Markersize',6);
    end
    ylabel(['d = ',num2str(l-1)])
    xlabel('x')
    hold off
end


