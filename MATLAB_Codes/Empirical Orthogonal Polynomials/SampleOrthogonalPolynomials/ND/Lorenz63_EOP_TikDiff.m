%% Solve 1 dof linear/quadratic undamped, unforced, oscillator problem
clc;
close all;
clear all;
addpath('./regtools','./utils')
rng(100);

%% Dynamical system parameters
%System parameters
sig = 10;
rho = 28;
beta = 8/3;
param = [sig,rho,beta];
n = 3; %# state variables;

%Number of samples after trimming the ends
trimratio = 0.1;%trimming ratio
sampleTime = 0.01; %Sampling time
m = 200; %Number of samples after trimming

%Time parameters
dt = 0.0001;
t0 = 0;
tf = t0 + m*sampleTime/(1-trimratio);

%Intial conditions
x0 = [-8;7;27];

tspanf = t0:dt:(tf-dt);

%State sapace model of a 2nd order linear differential equation
options = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,n));
[xf] = ode5(@(t,x) LorentzSys63(t,x,param),tspanf,x0');
x = xf;
t = tspanf';

%% Sample dynamical system
dsample = sampleTime/dt;
timeData = t(1:dsample:end);
Nsamples = length(timeData);

trimlow = round(trimratio/2*Nsamples);
trimupp = round((1-trimratio/2)*Nsamples);
trim = trimlow:trimupp;



%% True state variables
x1true = x(1:dsample:end,1);
x2true = x(1:dsample:end,2);
x3true = x(1:dsample:end,3);
xDataT = [x1true,x2true,x3true];

%% True state derivatives
dx1true = sig*(x2true-x1true);
dx2true = x1true.*(rho - x3true) - x2true;
dx3true = x1true.*x2true - beta*x3true;
dxDataT = [dx1true,dx2true,dx3true];

%% Noisy state derivatives
dxDataN = zeros(Nsamples,n);

%% NOISE MODELS

% SNR = 1000;%SNR is defined as (sqrt(1/N*sum_k(sk^2))/sigma_noise)^2 in terms of power
% sigma1 = sqrt(rms(x1true)^2/SNR);
% sigma2 = sqrt(rms(x2true)^2/SNR);
% sigma3 = sqrt(rms(x3true)^2/SNR);

sigmas = 0.0001;
Nsigmas = length(sigmas);
SNR1 = zeros(1,Nsigmas);
SNR2 = zeros(1,Nsigmas);
SNR3 = zeros(1,Nsigmas);

% SNR = [1e5,1e4,1e3,1e2,1e1,1];
% SNR = logspace(8,2,7);
% % SNR = 100;
% 
% Nsigmas = length(SNR);
% sigma1 = zeros(1,Nsigmas);
% sigma2 = zeros(1,Nsigmas);
% sigma3 = zeros(1,Nsigmas);


errorx_sig = zeros(Nsigmas,n);
errordx_sig = zeros(Nsigmas,n);
amplifdx_sig = zeros(Nsigmas,n);

lambda_min = eps;
lambda_max = 1;
lambdasLc = nlogspace(log(lambda_min),log(lambda_max),500);
Nlamb = length(lambdasLc);

errordxLcnoise = zeros(Nsigmas,n,Nlamb);
residualLcnoise = zeros(Nsigmas,n,Nlamb);
regularizerLcnoise = zeros(Nsigmas,n,Nlamb);

resCorner = zeros(Nsigmas, n);
regCorner = zeros(Nsigmas, n);


for j = 1:Nsigmas

sigma1 = sigmas(j);
sigma2 = sigmas(j);
sigma3 = sigmas(j);
SNR1(j) = (rms(x1true)/sigma1)^2;
SNR2(j) = (rms(x2true)/sigma2)^2;
SNR3(j) = (rms(x3true)/sigma3)^2;
    
x1noisy = x1true + sigma1*randn(Nsamples,1);

x2noisy = x2true + sigma2*randn(Nsamples,1);

x3noisy = x3true + sigma3*randn(Nsamples,1);

xDataN = [x1noisy,x2noisy,x3noisy];

errorxsn(1) = norm(x1noisy - x1true)/norm(x1true);
errorxsn(2) = norm(x2noisy - x2true)/norm(x2true);
errorxsn(3) = norm(x3noisy - x3true)/norm(x3true);

errorxs(1) = norm(x1noisy - x1true);
errorxs(2) = norm(x2noisy - x2true);
errorxs(3) = norm(x3noisy - x3true);

%% NUMERICAL DIFFERENTIATION

%%  Tikhonov regularization   
    epsilon = 0.01; % Corner point stopping criterion
    itr2_max = 50; % Maximum iterations to find the corner point
    pGS = (1+sqrt(5))/2; %Golden search parameter

    %Vector of ordinates
    tvec = timeData;
    tvecm = 0.5*(tvec(1:end-1) + tvec(2:end));
    
    errordxsn = zeros(1,n);
    errordxs = zeros(1,n);

    errordxLc = zeros(Nlamb,1);
    residualLc = zeros(Nlamb,1);
    regularizerLc = zeros(Nlamb,1);
    

for d = 1:n
    
    fprintf(['State: ', num2str(d),'\n'])
    
    b = xDataN(:,d);
    
    itr2 = 1;
    lambdas = [lambda_min,lambda_max,0,0];
    gap = 1;
    
    residual = zeros(1,4);
    regularizer = zeros(1,4);
    errordx = zeros(1,4);
    xis = zeros(1,4);
    etas = zeros(1,4);    
    
    %% Generate full Lcurve
    
    for l = 1:Nlamb
        l
        b = xDataN(:,d);
        lambda  = lambdasLc(l);
        [dxm,A,L,res,relres,normreg] = tik_diff_uniform(tvec,b,lambda);
        tdx = 0.5*(tvec(1:end-1) + tvec(2:end)); %Derivatives computed at the midpoints of t

        dx = pchip(tdx,dxm,tvec);%Interpolation to the grid points
        
        errordxLc(l) = norm(dxDataT(:,d) - dx)/norm(dx);
        residualLc(l) = res;
        regularizerLc(l) = normreg;
        
        errordxLcnoise(j,d,l) = errordxLc(l);
        residualLcnoise(j,d,l) = residualLc(l);
        regularizerLcnoise(j,d,l) = regularizerLc(l);
        
        
    end
%     
%     figLcurve = figure(1);
%     axLcurve = axes('Parent',figLcurve);
%     LcurvePlot = plot(axLcurve,log(residualLc),log(regularizerLc),'r.-');
%     ylabel('\eta = log(||Dx||_2)')
%     xlabel('\xi = log(||Ax - b||_2)')
%     grid on   
%     pause
%     indmin = find(errordxLc == min(errordxLc));
%     hold on
%     plot(axLcurve,log(residualLc(indmin)),log(regularizerLc(indmin)),'Color','g','Marker','.','MarkerSize',20)
%     drawnow
%      
%     figError = figure(2);
%     axError = axes('Parent',figError);
%     ErrorPlot = loglog(axError,lambdasLc,errordxLc,'r.-');
%     ylabel('log(||x - x^*||_2 / ||x^*||_2)')
%     xlabel('\lambda')
%     grid on
%     hold on
%     pause
    
    while (gap > epsilon || itr2 < itr2_max)
        
    % Solve BPDN for extremals lambda_min and lambda_max
    if itr2 == 1
        for s = 1:2 %Regularization parameter
        
            lambda = lambdas(s);
            fprintf(['Current lambda: ',num2str(lambda),'\n'])
        
            %Solve Tikhonov regularization problem
            [dxm,A,L,res,relres,normreg] = tik_diff_uniform(tvec,b,lambda);
             tdx = 0.5*(tvec(1:end-1) + tvec(2:end)); %Derivatives computed at the midpoints of t
             dx = pchip(tdx,dxm,tvec);%Interpolation to the grid points
             errordx(s) = norm(dxDataT(:,d) - dx)/norm(dxDataT(:,d));
             %Compute l2-norm of residual and l2-norm of weighted solution
             residual(s) = res;
             regularizer(s) = normreg;

             xis(s) = log(residual(s));
             etas(s) = log(regularizer(s));
        
        end

    lambdas(3) = exp((log(lambdas(2)) + pGS*log(lambdas(1)))/(1+pGS));%Store lambda 2
    lambdas(4) = exp(log(lambdas(1)) + log(lambdas(2)) - log(lambdas(3)));%Store lambda 3
    
    %Solve BPDN for intermediate lambdas 2,3
        for s = 3:4 %Regularization parameter
        
            lambda = lambdas(s);
            fprintf(['Current lambda: ',num2str(lambda),'\n'])
        
            %Solve Tikhonov regularization problem
            [dxm,A,L,res,relres,normreg] = tik_diff_uniform(tvec,b,lambda);
            tdx = 0.5*(tvec(1:end-1) + tvec(2:end)); %Derivatives computed at the midpoints of t
             dx = pchip(tdx,dxm,tvec);%Interpolation to the grid points
             %Compute relative error in the derivative
             errordx(s) = norm(dxDataT(:,d) - dx)/norm(dxDataT(:,d));
             %Compute l2-norm of residual and l2-norm of weighted solution
             residual(s) = res;
             regularizer(s) = normreg;

             xis(s) = log(residual(s));
             etas(s) = log(regularizer(s));
        
        end
    
            %Sort points (xi,eta) corresponding to each lambda in ascending order

            P = [xis(1),xis(2),xis(3),xis(4);etas(1),etas(2),etas(3),etas(4)];

            [lambdas, indx] = sort(lambdas);

            P = P(:,indx);%Sort points according to values of lambda in ascending order
       
     end%End of loop for the first iteration
    
    %% Compute curvatures
    
    %Compute coordimates of the 4 current points
    C2 = menger2(P(:,1),P(:,2),P(:,3));
    C3 = menger2(P(:,2),P(:,3),P(:,4));

    while C3 < 0 %Check if the curvature is negative and update values
        
         %Reassign maximum and interior lambdas and Lcurve points (Golden search interval)
         lambdas(4) = lambdas(3);
         P(:,4) = P(:,3);
         lambdas(3) = lambdas(2);
         P(:,3) = P(:,2);
        
         %Update interior lambda and interior point
         lambdas(2) = exp((log(lambdas(4)) + pGS*log(lambdas(1)))/(1+pGS));
        
         %Solve Tikhonov regularization problem for lambda2
         [dxm,A,L,res,relres,normreg] = tik_diff_uniform(tvec,b,lambdas(2));
         tdx = 0.5*(tvec(1:end-1) + tvec(2:end)); %Derivatives computed at the midpoints of t
         dx = pchip(tdx,dxm,tvec);%Interpolation to the grid points
        
         %Compute relative error in the derivative
         errordx = norm(dxDataT(:,d) - dx)/norm(dxDataT(:,d));
         %Compute l2-norm of residual and l2-norm of weighted solution
         residual = res;
         regularizer = normreg;

         xi = log(residual);
         eta = log(regularizer);
        
         P(:,2) = [xi;eta];
         C3 = menger2(P(:,2),P(:,3),P(:,4));
        
    end
    
    if C2 > C3 %Update values depending on the curvature at the new points
        
        display('Curvature C2 is greater than C3');
        lambdaC = lambdas(2);
        
        %Reassign maximum and interior lambdas and Lcurve points (Golden search interval)
        lambdas(4) = lambdas(3);
        P(:,4) = P(:,3);
        
        lambdas(3) = lambdas(2);
        P(:,3) = P(:,2);
        
        %Update interior lambda and interior point
        lambdas(2) = exp((log(lambdas(4)) + pGS*log(lambdas(1)))/(1+pGS));
        
        %Solve Tikhonov regularization problem for lambda2
         [dxm,A,L,res,relres,normreg] = tik_diff_uniform(tvec,b,lambdas(2));
         tdx = 0.5*(tvec(1:end-1) + tvec(2:end)); %Derivatives computed at the midpoints of t
         dx = pchip(tdx,dxm,tvec);%Interpolation to the grid points
        
         %Compute relative error in the derivative
         errordx = norm(dxDataT(:,d) - dx)/norm(dxDataT(:,d));
         %Compute l2-norm of residual and l2-norm of weighted solution
         residual = res;
         regularizer = normreg;

         xi = log(residual);
         eta = log(regularizer);
        
         P(:,2) = [xi;eta];

%          plot(axLcurve,xi,eta,'Color','k','Marker','o','MarkerSize',5,'LineStyle','none')
%          drawnow
%          plot(axError,lambdas(2),errordx,'Color','k','Marker','o','MarkerSize',5,'LineStyle','none')
%          drawnow
    else
        
        display('Curvature C3 is greater than C2');
        lambdaC = lambdas(3);
        
        %Reassign maximum and interior lambdas and Lcurve points (Golden search interval)
        lambdas(1) = lambdas(2);
        P(:,1) = P(:,2);
        
        lambdas(2) = lambdas(3);
        P(:,2) = P(:,3);
        
        %Update interior lambda and interior point
        lambdas(3) = exp(log(lambdas(1)) + log(lambdas(4)) - log(lambdas(2)));
        
        %Solve Tikhonov regularization problem for lambda3
         [dxm,A,L,res,relres,normreg] = tik_diff_uniform(tvec,b,lambdas(3));
         tdx = 0.5*(tvec(1:end-1) + tvec(2:end)); %Derivatives computed at the midpoints of t
         dx = pchip(tdx,dxm,tvec);%Interpolation to the grid points
        
         %Compute relative error in the derivative
         errordx = norm(dxDataT(:,d) - dx)/norm(dxDataT(:,d));
         %Compute l2-norm of residual and l2-norm of weighted solution
         residual = res;
         regularizer = normreg;

         xi = log(residual);
         eta = log(regularizer);
        
         P(:,3) = [xi;eta];

%          plot(axLcurve,xi,eta,'Color','k','Marker','o','MarkerSize',5,'LineStyle','none')
%          drawnow
%          plot(axError,lambdas(3),errordx,'Color','k','Marker','o','MarkerSize',5,'LineStyle','none')
%          drawnow
        
    end
    
    %Compute relative gap
    gap = (lambdas(4) - lambdas(1))/lambdas(4);
    lambda_itr2(itr2) = lambdaC;
    itr2 = itr2 + 1;    
    
    end
%     pause
%     close(figLcurve)
%     close(figError)
    
    %% Pick the corner
    lambda_corner = lambdaC;
    display(['Optimal lambda: ',num2str(lambda_corner)])

        
        %Solve Tikhonov regularization problem for optimal lambda
         [dxm,A,L,res,relres,normreg] = tik_diff_uniform(tvec,b,lambda_corner);
         resCorner(j,d) = res;
         regCorner(j,d) = normreg;
         tdx = 0.5*(tvec(1:end-1) + tvec(2:end)); %Derivatives computed at the midpoints of t
         dx = pchip(tdx,dxm,tvec);%Interpolation to the grid points
         dxDataN(:,d) = dx';
         errordxsn(d) = norm(dxDataT(:,d) - dx)/norm(dxDataT(:,d));
         errordxs(d) = norm(dxDataT(:,d) - dx);

         display(['Derivative error for state ',num2str(d),' = ',num2str(errordxs(d))])
%          lambda_corner
%          pause
%          close all
end

display(['Error in states: ',num2str(errorxsn)])
display(['Error in derivatives: ',num2str(errordxsn)])
display(['Error amplification: ',num2str(errordxs./errorxs)])
% pause
% close

errorx_sig(j,:) = errorxsn
errordx_sig(j,:) = errordxsn
amplifdx_sig(j,:) = errordxs./errorxs

Lorenz63Data(j).xtrue = xDataT(trim,:);
Lorenz63Data(j).dxtrue = dxDataT(trim,:);
Lorenz63Data(j).xnoisy = xDataN(trim,:);
Lorenz63Data(j).dxnoisy = dxDataN(trim,:);
Lorenz63Data(j).time = timeData(trim);
% Lorenz63Data(j).sigma = [sigma1(j), sigma2(j),sigma3(j)];
Lorenz63Data(j).sigma = sigmas;
Lorenz63Data(j).SNR = [SNR1(j), SNR2(j),SNR3(j)];
Lorenz63Data(j).paramsLabel = ['sigma, rho, beta'];
Lorenz63Data(j).paramsValue = [sig,rho,beta];

% pause

end

xDataN = Lorenz63Data(1).xnoisy;
dxDataN = Lorenz63Data(1).dxnoisy;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% GENERATE BASIS MATRIX
deg = 3;
xi = xDataN';
m = size(xDataN,1);

%% Generate multi index array (graded lexicographic order)
%Dimension of the basis
dim = nchoosek(n + deg,deg);
%Construct the multi index array

if n == 1
    alpha(:,1) = 0:1:deg;
else
    alpha = nd_multi_index_poly(n,deg);
end

%Generate multidimensional, non-orthogonal polynomial basis matrix
X = zeros(m,dim);
for j = 1:dim
            alphajs = repmat(alpha(j,:),m,1);
            X(:,j) = prod(xi'.^(alphajs),2);
end

%% ORTHOGONALIZATION WITH RESPECT TO THE SAMPLE MEASURE 
[Qmgs,Rmgs] = MGS_nD_samples(X);
G_MGS = 1/m*Qmgs'*Qmgs;

%Check orthogonality (Mutual coherence)
A = zeros(dim);
for c = 1:dim
    A(:,c) = G_MGS(:,c)/norm(G_MGS(:,c));
end
MC = max(max(abs(A - diag(diag(A)))))
%Check orthonormality
e_OrthN = norm(G_MGS - eye(dim),'f')

%Compute condition number
fprintf('Condition number of the polynomial basis:')
cond(X)
fprintf('Condition number of the orthogonal basis:')
cond(Qmgs)

%% ORDINARY LEAST SQUARES TO OBTAIN THE PARAMETERS

Xi = X\dxDataN;

XiO = Qmgs\dxDataN;

