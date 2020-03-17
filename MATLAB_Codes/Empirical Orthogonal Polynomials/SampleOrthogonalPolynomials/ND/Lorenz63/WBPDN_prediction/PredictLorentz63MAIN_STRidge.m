%% Identification of the Lorenz63 system
clc;
close all;
clear all;
addpath('./utils','./Data')

%%%%% WL1min identified model at iteration 0 %%%%% 
% load('Xi_it0_Lorenz63')
% XiI = Xi_it0;

%%%%% WL1min identified model at iteration 5 %%%%% 
load('STRidgeResultsMean')
noise = 4;
XiI = squeeze(Xi_STR_noise_mean(:,:,noise));

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
tf = 10; %Final time

%Intial conditions
x0 = [-8;7;27];

%Time span
% tspan = t0:dt:tf;
tspan = t0:tf;


%% Generate true model
% options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
options = [];
[tT, xT] = ode45(@(t,x) LorentzSys63(t,x,param),tspan,x0,options);

%% Generate identified model
[tI_STR, xI_STR] = ode45(@(t,x)predictLorenz63(t,x,XiI,plyord),tspan,x0,options);

%% Plot data (true vs identified states)
figure(1);
suptitle('Exact vs predicted Lorenz63 states')
subplot(3,1,1)
plot(tT,xT(:,1),'g-',tI_STR,xI_STR(:,1),'k-')
xlabel('Time');
ylabel('state x(t)');
grid on
legend('Exact x','Predicted x')

subplot(3,1,2)
plot(tT,xT(:,2),'g-',tI_STR,xI_STR(:,2),'k-')
xlabel('Time');
ylabel('state y(t)');
grid on
legend('Exact y','Predicted y')

subplot(3,1,3)
plot(tT,xT(:,3),'g-',tI_STR,xI_STR(:,3),'k-')
xlabel('Time');
ylabel('state z(t)');
grid on
legend('Exact z','Predicted z')

%% Plot data (true vs identified states)
figure(2);
suptitle('Exact vs predicted Lorenz63 models')
plot3(xT(:,1),xT(:,2),xT(:,3),'g-')
hold on
plot3(xI_STR(:,1),xI_STR(:,2),xI_STR(:,3),'k-')
grid on
axis('equal')
xlabel('x(t)');
ylabel('y(t)');
zlabel('z(t)');
legend('Exact model','Identified model')

save('PredictedLorenz63Data_STRidge','xT','xI_STR','tT')
