%% Identification of the Lorenz63 system
clc;
close all;
clear all;
addpath('./utils','./Data')

%%%%% WL1min identified model at iteration 0 %%%%% 
% load('Xi_it0_Lorenz63')
% XiI = Xi_it0;

%%%%% WL1min identified model at iteration 5 %%%%%
filename = 'Lorenz63_3Dplot';
load('Lorenz63TikhonovNumDiff200')
noise = 4;
xDataN = Lorenz63Data(noise).xnoisy;
timeN = Lorenz63Data(noise).time;
%System parameters
sigma = 10;
rho = 28;
beta = 8/3;
param = [sigma,rho,beta];
n = 3; %# state variables;
plyord = 3;
usesine = 0;

%Time parameters
dt = 0.0001; %Time step
t0 = 0; %Initial time
tf = 50; %Final time

%Intial conditions
x0 = [-8;7;27];

%Time span
tspan = t0:dt:tf;

%% Generate true model
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[tT, xT] = ode45(@(t,x) LorentzSys63(t,x,param),tspan,x0,options);

%% Plot data (true vs identified states)
figure(1);
suptitle('Exact vs noisy Lorenz63 states')
subplot(3,1,1)
plot(tT,xT(:,1),'g-',timeN,xDataN(:,1),'k-')
xlabel('Time');
ylabel('state x(t)');
grid on
legend('Exact x','Data x')

subplot(3,1,2)
plot(tT,xT(:,2),'g-',timeN,xDataN(:,2),'k-')
xlabel('Time');
ylabel('state y(t)');
grid on
legend('Exact y','Data y')

subplot(3,1,3)
plot(tT,xT(:,3),'g-',timeN,xDataN(:,3),'k-')
xlabel('Time');
ylabel('state z(t)');
grid on
legend('Exact z','Data z')

%% Plot data (true vs identified states)
figure(2);
plot3(xT(:,1),xT(:,2),xT(:,3),'Color',1/255*[200,200,200],'LineStyle','-','LineWidth',2)
hold on
plot3(xDataN(:,1),xDataN(:,2),xDataN(:,3),'r.','MarkerSize',15)
grid on
axis('square')
view([48.9 26]);
xlabel('x(t)');
ylabel('y(t)');
zlabel('z(t)');
legend('Exact trajectory','Measured trajectory')

Lorenz63_3D_plot(xT(:,1),xT(:,2),xT(:,3),xDataN(:,1),xDataN(:,2),xDataN(:,3))
print(filename,'-depsc')
