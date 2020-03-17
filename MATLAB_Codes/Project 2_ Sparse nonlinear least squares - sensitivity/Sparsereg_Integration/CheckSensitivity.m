%% CHECK SENSITIVITY ANALYSIS
clc;
close all;
clear all;
addpath('./Solvers','./utils')

omegap = 1;
epsip = 5;
xip = 0.05;

% Setup the basis
n=2;
poldeg = 4;
index_pc = nD_polynomial_array(n,poldeg);
p = size(index_pc,1);

%% Initial guess on parameter vector
%True solution vector
XiT = zeros(p,n);
XiT(3,1) = 1;
XiT(3,2) = -2*omegap*xip;
XiT(2,2) = - omegap^2;
XiT(7,2) = - omegap^2*epsip;

%Intial guess (perturbed)
a = -1;
b = 1;
sc = 0.1;%scaling factor of the perturbation for the initial guess
r = a + (b-a).*rand(p,n);
Xi0 = XiT + sc*r;

Xi = Xi0;

%Time parameters.
sampleTime = 0.01; %Sampling time
m = 200; %Number of samples after trimming
dt = 0.0001;
t0 = 0;
tf = t0 + m*sampleTime;
dsample = sampleTime/dt;
timeData = t0:sampleTime:tf;


%% Initial conditions
x0 = [0;1];

x0m = x0;
S0 = zeros(n,p);
s0 = S0(:);
xs0 = [x0;s0];
nds = length(xs0);
dxS = zeros(n,p+1,length(timeData));
dxST = zeros(n,p+1,length(timeData));



%Predict state with true parameter vector
options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,nds));
[tdxsT,dxsT] = ode45(@(t,x)IntegrateModel(t,x,XiT,index_pc),timeData,xs0,options);
dxsT = dxsT';
for t = 1:size(dxsT,2)
    dxST(:,:,t) = reshape(dxsT(:,t),n,p+1);
    xpT = squeeze(dxST(:,1,:))';
    SpT = dxST(:,2:end,:);
end

SpT1 = squeeze(SpT(1,:,:))';
SpT2 = squeeze(SpT(2,:,:))';


%Compute solution error and const function for the initial guess (perturbed
%parameter vector)
options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,nds));
[tdxs,dxs] = ode45(@(t,x)IntegrateModel(t,x,Xi,index_pc),timeData,xs0,options);
dxs = dxs';
for t = 1:size(dxs,2)
    dxS(:,:,t) = reshape(dxs(:,t),n,p+1);
    xp = squeeze(dxS(:,1,:))';
    Sp = dxS(:,2:end,:);
end
Sp1 = squeeze(Sp(1,:,:))';
Sp2 = squeeze(Sp(2,:,:))';
    
 %Compute sensitivity by finite differencing (perturbation of parameters)
 for i = 1:length(timeData)
     for j=1:p
     	Sp1_FD(i,j) = (xp(i,1)-xpT(i,1))/(Xi(j,1)-XiT(j,1));
        Sp2_FD(i,j) = (xp(i,2)-xpT(i,2))/(Xi(j,2)-XiT(j,2));
     end
 end
 
 figure
 subplot(2,1,1)
 for k = 1:p
     plot(timeData,SpT1(:,k) - Sp1_FD(:,k))
     hold on
 end
xlabel('time')
ylabel('sensitivity error of state 1')
subplot(2,1,2)
 for k = 1:p
     plot(timeData,SpT2(:,k) - Sp2_FD(:,k))
     hold on
 end
 xlabel('time')
 ylabel('sensitivity error of state 2')
        
        
        
        
        
        
        
        
        
        
        
        
        