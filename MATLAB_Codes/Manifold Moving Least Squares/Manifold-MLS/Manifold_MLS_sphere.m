%% IMPLEMENTATION OF THE ALGORITHM IN "Approximation of functions over manifolds: A Moving Least Squares Approach"
clc;
close all;
clear all;

%% GENERATE DATA
%Number of samples
N = 10000; 
%Independent variable
rho = 1;
theta = pi*rand(1,N);
phi = 2*pi*rand(1,N);

[Xm,Ym,Zm] = sphere(50);
%1D helix Manifold on a R^3 ambient space
x = rho.*sin(theta).*cos(phi);
y = rho.*sin(theta).*sin(phi);
z = rho.*cos(theta);
rt = [x;y;z];

%Add noise (AWGN)
sigma = 0*0.001;
xn = x + sigma*randn(1,N);
yn = y + sigma*randn(1,N);
zn = z + sigma*randn(1,N);

rn = [xn;yn;zn];

%Plot manifold and noisy samples
surf(Xm,Ym,Zm,'EdgeColor','none','FaceAlpha',0.5)
hold on
plot3(xn,yn,zn,'r.')
axis equal
pause
close
%% Algorithm 1
% Dimension of the ambient space and embedded manifold 
n = 3;
d = 2;
%Define R to be an n x N matrix whose columns are rni
R = rn;
%Parameters
indx = randi(N);
r = rt(:,indx) + sigma/2*randn(n,1);
epsilon = 1e-6;
kh = 2;
[q,H,convergence] = localSubspace(R,r,epsilon,kh,d);

figure
plot3(x,y,z,'k')
hold on
plot3(r(1),r(2),r(3),'rx','MarkerSize',10)
plot3(q(1),q(2),q(3),'go','MarkerSize',6)
quiver3(q(1),q(2),q(3),H(1),H(2),H(3));
legend('Manifold','Approximation point','Local subspace origin','Local subspace')
% axis('equal');

figure;
semilogy(1:length(convergence),convergence,'k');
xlabel('iteration #');
ylabel('log( 2-norm(q-qprev) )')





