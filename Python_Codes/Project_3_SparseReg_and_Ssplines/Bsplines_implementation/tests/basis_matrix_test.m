%% TEST - CONSTRUCTION OF B-SPLINE BASIS MATRIX

clc;
close all;
clear all;

knots = [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1];
N = 3;%Interior knots
d = 3;%polynomial degree
M = d + 1;%spline order

x = 0:0.01:1;

[B,x] = bspline_basismatrix(M,knots,x);

for i = 1:N+M
    plot(x,B(:,i));
    hold on
end

%% Fitting test
xx = 0:0.01:1;
y = 2*sin(10*xx') + xx'.^2;
yy = y + 0.1*randn(length(xx),1);
knots = 0:0.1:1;


[Bfit,x] = bspline_basismatrix(M,knots,xx);
Q = Bfit' * yy;
C = (Bfit'*Bfit)\Q;

y_spline = Bfit*C;
figure(2);
plot(xx,yy,'o-')
hold on
plot(xx,y_spline,'k')
