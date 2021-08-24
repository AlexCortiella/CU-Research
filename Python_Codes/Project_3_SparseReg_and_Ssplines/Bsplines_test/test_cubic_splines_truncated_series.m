%% Example of truncated cubic splines
clc;
close all;
clear all;

%% Generate data
x = 0:0.01:1;
y = sin(10*x);

m_samples = length(x);

plot(x,y,'ro')
pause
close;

%% Cubic splines setup
d = 3; %degree of the polynomial
M = d+1; %order of spline
n_knots = 10;

% knots = [0.25, 0.5, 0.75];%interior knots
knots = linspace(0,1,n_knots);%interior knots

K = length(knots);

n_basis = K+M;

%% Generate truncated power series for splines
B = zeros(m_samples, n_basis);
for i = 1:m_samples
    for j = 1:n_basis
        
        if j <= M
        
            B(i,j) = x(i).^(j-1);
        else
           
            B(i,j) = func_plus(x(i) - knots(j-M)).^d;
        end
    end
end

%% Solve for coefficients
theta = B\y';

y_spline = B*theta;

%% Verify model

figure;
plot(x,y,'ro')
hold on
plot(x,y_spline,'b')
plot(knots,0,'Marker','s','MarkerFaceColor','k','Markersize',8);


