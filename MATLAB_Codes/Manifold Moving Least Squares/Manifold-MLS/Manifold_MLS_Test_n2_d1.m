%% IMPLEMENTATION OF THE ALGORITHM IN "Approximation of functions over manifolds: A Moving Least Squares Approach"
clc;
close all;
clear all;

%% GENERATE DATA
%Independent variable
t0 = -2*pi;
tf = 2*pi;
dt = 0.01;
t = t0:dt:tf;
%Number of samples
N = length(t); 

%1D helix Manifold on a R^3 ambient space
x = linspace(-1,1,N);
y = 2*x;
rt = [x;y];

%Add noise (AWGN)
sigma = 0.01;
xn = x + sigma*randn(1,N);
yn = y + sigma*randn(1,N);

rn = [xn;yn];

%Plot manifold and noisy samples
plot(x,y,'k')
hold on
plot(xn,yn,'r.')
pause
close
%% Algorithm 1
% Dimension of the ambient space and embedded manifold 
n = 2;
d = 1;
%Define R to be an n x N matrix whose columns are rni
R = rn;
%Parameters
indx = randi(N);
% eta = sigma;
eta = 0.05;
r = rt(:,indx) + 0.8*eta*randn(n,1);
figure
plot(x,y,'k')
hold on
plot(xn,yn,'r.')
plot(r(1),r(2),'k.','MarkerSize',12)
pause
close
epsilon = 1e-6;
kh = 2;
[q,H,convergence] = localSubspace(R,r,epsilon,kh,d);

figure
plot(x,y,'k')
hold on
plot(r(1),r(2),'rx','MarkerSize',10)
plot(q(1),q(2),'go','MarkerSize',6)
quiver(q(1),q(2),H(1),H(2));
legend('Manifold','Approximation point','Local subspace origin','Local subspace')
% axis('equal');

figure;
semilogy(1:length(convergence),convergence,'k');
xlabel('iteration #');
ylabel('log( 2-norm(q-qprev) )')





