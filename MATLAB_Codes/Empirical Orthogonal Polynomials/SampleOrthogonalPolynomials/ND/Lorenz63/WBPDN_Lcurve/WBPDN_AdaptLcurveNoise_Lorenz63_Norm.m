%% SPARSE IDENTIFICATION OF NONLINEAR DYNAMICAL SYSTEMS

clear all, close all, clc
addpath('./Solvers','./utils');

%% Load data
load('Lorenz63DataStruc200');

% Order of expansion
p = 3;
% p = 4;
Nsamp_total = 200;
% num_itr = 5;
num_itr = 3;


%Noise levels
Nsigmas = length(Lorenz63Data);

% True coefficient vector in monomial basis
Nb = nchoosek(p+3,3);
XiT = zeros(Nb,3);
sig = Lorenz63Data(1).paramsValue(1);
rho = Lorenz63Data(1).paramsValue(2);
beta = Lorenz63Data(1).paramsValue(3);

sigmas = Lorenz63Data(1).sigma;


XiT(2,1) = -sig;
XiT(2,2) = rho;
XiT(3,1) = sig;
XiT(3,2) = -1;
XiT(4,3) = -beta;
XiT(6,3) = 1;
XiT(8,2) = -1;

% Nsigmas = 1;
%Preallocate variables
tr_errLc_noise = zeros(Nsigmas,3,num_itr,200);
sol_errLc_noise = zeros(Nsigmas,3,num_itr,200);
sol_l1_normLc_noise = zeros(Nsigmas,3,num_itr,200);

sol_l1_normCorner = zeros(Nsigmas,3,num_itr+1);
tr_errsCorner = zeros(Nsigmas,3,num_itr+1);

Xi_noise = zeros(Nsigmas,3,Nb)

for j = 4:Nsigmas

%Gather data

%True data
xDataT = Lorenz63Data(j).xtrue(1:Nsamp_total,:);
dxDataT = Lorenz63Data(j).dxtrue(1:Nsamp_total,:);
tDataT = Lorenz63Data(j).time(1:Nsamp_total,:);
dtDataT = mean(tDataT(2:end) - tDataT(1:end-1));

%Noisy data
xData = Lorenz63Data(j).xnoisy(1:Nsamp_total,:);
dxData = Lorenz63Data(j).dxnoisy(1:Nsamp_total,:);


% Setup the basis
Ndofs = size(xDataT,2);
index_pc = nD_polynomial_array(Ndofs,p);
Nbasis = size(index_pc,1);

%Allocate identified coefficients
Xi = zeros(Nbasis,Ndofs);

%% START L-Curve

%Define sigma and p ranges to loop over
%Define sigmas min sigma*|x|_1 + |Ax - b|_2 

lambda_itr = zeros(num_itr+1);
epsilon = 0.01; % Corner point stopping criterion
itr2_max = 50; % Maximum iterations to find the corner point
pGS = (1+sqrt(5))/2; %Golden search parameter

%Define penalization parameter (exponents of the weighting matrix) 
q = 2;
eps_w = 0.0001;
coefferr = zeros(num_itr+1,1);% NOTE: the first row represents the solution error at 0th iteration

%Build dictionary for the entire data set
C = zeros(Nsamp_total,Nbasis);
for isim = 1:Nsamp_total
    crow = piset_monomial(xData(isim,:),index_pc);
    C(isim,:) = crow(1:Nbasis);
end

%Normalize columns
Ccol_norm = (sqrt(sum(C.*C,1)))';
Wn = diag(1./Ccol_norm); %Normalization matrix 
Cn = C * Wn; %Column-normalized basis matrix

cond(C)
cond(Cn)

%% Zeroth iteration (no weighting)
for d = 1:Ndofs

    fprintf(['Degree of freedom: ', num2str(d),'\n'])
    
    b = dxData(:,d);
    Xitrue = XiT(:,d);
    
    %Define regularization parameter range
    lambda_min = eps;
%     lambda_max = max(abs(C'*b));
    lambda_max = 100;
    
    lambdasLc = nlogspace(log(lambda_min),log(lambda_max),200);
    NlambdasLc = length(lambdasLc);

    tr_errLc = zeros(NlambdasLc,1);
    sol_errLc = zeros(NlambdasLc,1);
    sol_l1_normLc = zeros(NlambdasLc,1);
    xiLc = zeros(NlambdasLc,1);
    etaLc = zeros(NlambdasLc,1);
    
    itr2 = 1;
    lambdas = [lambda_min,lambda_max,0,0];
    gap = 1;
    
    tr_errs = zeros(1,4);
    sol_errs = zeros(1,4);
    sol_l1_norms = zeros(1,4);
    xis = zeros(1,4);
    etas = zeros(1,4);
    
    %% Generate full Lcurve
    
    for s = 1:NlambdasLc %Regularization parameter
        
        lambdaLc = lambdasLc(s);
        fprintf(['Current sigma: ',num2str(lambdaLc),'\n'])
        
        %Solve training set
        
        y_tilde = SolveBP(Cn,b,Nbasis,100000,lambdaLc,1e-11);
        Xi = Wn*y_tilde;%De-normalize coefficients
                
        %Validate data
        tr_errLc(s) = norm(C*Xi - b);
        sol_errLc(s) = norm(Xi-Xitrue)/norm(Xitrue);
        sol_l1_normLc(s) = norm(y_tilde,1);
        
        xiLc(s) = log(tr_errLc(s));
        etaLc(s) = log(sol_l1_normLc(s));
        
    end
    
    %Normalize between -1 and 1
    CshxiLc = 1 - 2/(xiLc(end) - xiLc(1))*xiLc(end);
    CscxiLc = 2/(xiLc(end) - xiLc(1));
    CshetaLc = 1 - 2/(etaLc(1) - etaLc(end))*etaLc(1);
    CscetaLc = 2/(etaLc(1) - etaLc(end));
    
    xihatLc = CshxiLc + CscxiLc*xiLc;
    etahatLc = CshetaLc + CscetaLc*etaLc;
   
    
    tr_errLc_noise(j,d,1,:) = tr_errLc;
    sol_errLc_noise(j,d,1,:) = sol_errLc;
    sol_l1_normLc_noise(j,d,1,:) = sol_l1_normLc;
    
        
    figLcurve = figure(1);
    axLcurve = axes('Parent',figLcurve);
    LcurvePlot = plot(axLcurve,xihatLc,etahatLc,'r.-');
    ylabel('\eta = log(||x||_1)')
    xlabel('\xi = log(||Ax - b||_2)')
    grid on   

    indmin = find(sol_errLc == min(sol_errLc));
    hold on
    plot(axLcurve,xihatLc(indmin),etahatLc(indmin),'Color','g','Marker','.','MarkerSize',20)
    drawnow
    
     
    figError = figure(2);
    axError = axes('Parent',figError);
    ErrorPlot = loglog(axError,lambdasLc,sol_errLc,'r.-');
    ylabel('log(||x - x^*||_2 / ||x^*||_2)')
    xlabel('\lambda')
    grid on
    hold on
    
    while (gap > epsilon || itr2 < itr2_max)
        
    % Solve BPDN for extremals lambda_min and lambda_max
    if itr2 == 1
    for s = 1:2 %Regularization parameter
        
        lambda = lambdas(s);
        fprintf(['Current lambda: ',num2str(lambda),'\n'])
        
        %Solve for lambda_min  and lambda_max
        y_tilde = SolveBP(Cn,b,Nbasis,100000,lambda,1e-11);
        Xi = Wn*y_tilde;%De-normalize coefficients
        %Compute l2-norm of residual and l1-norm of solution
        tr_errs(s) = norm(C*Xi - b);%/norm(dxData(:,d));
        sol_errs(s) = norm(Xi-Xitrue)/norm(Xitrue);
        sol_l1_norms(s) = norm(y_tilde,1);

        
        xis(s) = log(tr_errs(s));
        etas(s) = log(sol_l1_norms(s));
        
    end

    lambdas(3) = exp((log(lambdas(2)) + pGS*log(lambdas(1)))/(1+pGS));%Store lambda 2
    lambdas(4) = exp(log(lambdas(1)) + log(lambdas(2)) - log(lambdas(3)));%Store lambda 3

    %Solve BPDN for intermediate lambdas 2,3
    for s = 3:4 %Regularization parameter
        
        lambda = lambdas(s);
        fprintf(['Current lambda: ',num2str(lambda),'\n'])
        
        %Solve for lambda_min  and lambda_max
        y_tilde = SolveBP(Cn,b,Nbasis,100000,lambda,1e-11);
        Xi = Wn*y_tilde;%De-normalize coefficients
        
        %Compute l2-norm of residual and l1-norm of solution
        tr_errs(s) = norm(C*Xi - b);%/norm(dxData(:,d));
        sol_errs(s) = norm(Xi-Xitrue)/norm(Xitrue);
        sol_l1_norms(s) = norm(y_tilde,1);
        
        xis(s) = log(tr_errs(s));
        etas(s) = log(sol_l1_norms(s));
        
    end
    
     %Normalize between -1 and 1
    Cshxi = 1 - 2/(xis(2) - xis(1))*xis(2);
    Cscxi = 2/(xis(2) - xis(1));
    Csheta = 1 - 2/(etas(1) - etas(2))*etas(1);
    Csceta = 2/(etas(1) - etas(2));
    xishat = Cshxi + Cscxi*xis;
    etashat = Csheta + Csceta*etas;
    
    %Sort points (xi,eta) corresponding to each lambda in ascending order
    %(lmin < l2 < l3 < lmax)

%     P = [xis(1),xis(2),xis(3),xis(4);etas(1),etas(2),etas(3),etas(4)];
    P = [xishat(1),xishat(2),xishat(3),xishat(4);etashat(1),etashat(2),etashat(3),etashat(4)];

    [lambdas, indx] = sort(lambdas);

    P = P(:,indx);%Sort points according to values of lambda in ascending order
       
    plot(axLcurve,P(1,:),P(2,:),'Color','b','Marker','o')
    drawnow
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
        
        %Solve for lambda2
        y_tilde = SolveBP(Cn,b,Nbasis,100000,lambdas(2),1e-11);
        Xi = Wn*y_tilde;%De-normalize coefficients
        
        %Compute l2-norm of residual and l1-norm of solution
        tr_err = norm(C*Xi - b);%/norm(dxData(:,d));
        sol_err = norm(Xi-Xitrue)/norm(Xitrue);
        sol_l1_norm = norm(y_tilde,1);

        
        xi = log(tr_err);
        eta = log(sol_l1_norm);
        
        xihat = Cshxi + Cscxi*xi;
        etahat = Csheta + Csceta*eta;
   
        %Normalize curve
         P(:,2) = [xihat;etahat];
        
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
        
        %Solve for lambda2
        y_tilde = SolveBP(Cn,b,Nbasis,100000,lambdas(2),1e-11);
        Xi = Wn*y_tilde;%De-normalize coefficients
        
        %Compute l2-norm of residual and l1-norm of solution
        tr_err = norm(C*Xi - b);%/norm(dxData(:,d));
        sol_err = norm(Xi-Xitrue)/norm(Xitrue);
        sol_l1_norm = norm(y_tilde,1);

        xi = log(tr_err);
        eta = log(sol_l1_norm);
        
        xihat = Cshxi + Cscxi*xi;
        etahat = Csheta + Csceta*eta;

        P(:,2) = [xihat;etahat];

        plot(axLcurve,xihat,etahat,'Color','k','Marker','o','MarkerSize',5,'LineStyle','none')
        drawnow
        plot(axError,lambdas(2),sol_err,'Color','k','Marker','o','MarkerSize',5,'LineStyle','none')
        drawnow

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
        
        %Solve for lambda3
        y_tilde = SolveBP(Cn,b,Nbasis,100000,lambdas(3),1e-11);
        Xi = Wn*y_tilde;%De-normalize coefficients
        
        %Compute l2-norm of residual and l1-norm of solution
        tr_err = norm(C*Xi - b);%/norm(dxData(:,d));
        sol_err = norm(Xi-Xitrue)/norm(Xitrue);
        sol_l1_norm = norm(y_tilde,1);

        xi = log(tr_err); 
        eta = log(sol_l1_norm);
        
        xihat = Cshxi + Cscxi*xi;
        etahat = Csheta + Csceta*eta;
        
        P(:,3) = [xihat;etahat];
        
        plot(axLcurve,xihat,etahat,'Color','k','Marker','o','MarkerSize',5,'LineStyle','none')
        drawnow
        plot(axError,lambdas(2),sol_err,'Color','k','Marker','o','MarkerSize',5,'LineStyle','none')
        drawnow

    end
    
    %Compute relative gap
    gap = (lambdas(4) - lambdas(1))/lambdas(4);
    lambda_itr2(itr2) = lambdaC;
    itr2 = itr2 + 1;
%     pause
    end
    
    %% Pick the corner
    lambda_corner = lambdaC;

    lambda_itr(1) = lambda_corner;
    fprintf(['Optimal sigma: ',num2str(lambda_corner)],'\n\n')

    % Solve L1 minimization with optimal sigma
    y_tilde = SolveBP(Cn,b,Nbasis,100000,lambda_corner,1e-11);
    Xi = Wn*y_tilde;%De-normalize coefficients
    coefferr(1) = norm(Xi - Xitrue)/norm(Xitrue);
    coefferr_noise(j,d,1) = coefferr(1);
    fprintf(['Coefficient error ',' = ',num2str(coefferr(1)),'\n'])
    % End of initial condition
    sol_l1_normCorner(j,d,1) = norm(y_tilde,1);
    tr_errsCorner(j,d,1) = norm(C*Xi - b);
    lambda_corner
    fprintf('Zeroth iteration: \n')
    Xi
    pause
    
    fprintf('START ITERATIVE LOOP... \n\n')

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% START WEIGHTED ITERATIONS

    %Restart variables
    tr_errs = zeros(1,4);
    sol_errs = zeros(1,4);
    sol_l1_norms = zeros(1,4);
    xis = zeros(1,4);
    etas = zeros(1,4);
        
    for itr=1:num_itr
        
        %Apply weighting
        w = 1./(abs(Xi).^q+eps_w);%Compute weights
        Ww = diag(1./w); %Compute weighting/penalization matrix
        Cnw = Cn*Ww; %Weight Cn

        for s = 1:NlambdasLc %Regularization parameter
        
        lambdaLc = lambdasLc(s);
        fprintf(['Current lambda: ',num2str(lambdaLc),'\n'])
        
        %Solve training set
        
        y_tilde = SolveBP(Cnw,b,Nbasis,100000,lambdaLc,1e-11);
        y_hat = Wn*y_tilde;
        Xi = Ww*y_hat;%De-normalize coefficients
                
        %Validate data
        tr_errLc(s) = norm(C*Xi - b);%/norm(dxData(:,d));
        sol_errLc(s) = norm(Xi-Xitrue)/norm(Xitrue);
        sol_l1_normLc(s) = norm(y_tilde,1);
        
        xiLc(s) = log(tr_errLc(s));
        etaLc(s) = log(sol_l1_normLc(s));
        
        end
        
        %Normalize between -1 and 1
        CshxiLc = 1 - 2/(xiLc(end) - xiLc(1))*xiLc(end);
        CscxiLc = 2/(xiLc(end) - xiLc(1));
        CshetaLc = 1 - 2/(etaLc(1) - etaLc(end))*etaLc(1);
        CscetaLc = 2/(etaLc(1) - etaLc(end));
    
        xihatLc = CshxiLc + CscxiLc*xiLc;
        etahatLc = CshetaLc + CscetaLc*etaLc;
        
        tr_errLc_noise(j,d,itr+1,:) = tr_errLc;
        sol_errLc_noise(j,d,itr+1,:) = sol_errLc;
        sol_l1_normLc_noise(j,d,itr+1,:) = sol_l1_normLc;
        
        figLcurveIter = figure(3);
        axLcurveIter = axes('Parent',figLcurveIter);
        LcurvePlot = plot(axLcurveIter,xihatLc,etahatLc,'r.-');
        ylabel('\eta = log(||x||_1)')
        xlabel('\xi = log(||Ax - b||_2)')
        grid on   

        indmin = find(sol_errLc == min(sol_errLc));
        hold on
        plot(axLcurveIter,xihatLc(indmin),etahatLc(indmin),'Color','g','Marker','.','MarkerSize',20)
        drawnow

        figErrorIter = figure(4);
        axErrorIter = axes('Parent',figErrorIter);
        ErrorPlotIter = loglog(axErrorIter,lambdasLc,sol_errLc,'r.-');
        ylabel('log(||x - x^*||_2 / ||x^*||_2)')
        xlabel('\lambda')
        grid on
        hold on

%         Reset parameters   
        itr2 = 1;
        gap = 1;
        lambdas = [lambda_min,lambda_max,0,0];
        
        while (gap > epsilon || itr2 < itr2_max)
        
        %Solve BPDN for extremals lambda_min and lambda_max
        if itr2 == 1
            for s = 1:2 %Regularization parameter
        
            lambda = lambdas(s);
            fprintf(['Current lambda: ',num2str(lambda),'\n'])
        
            %Solve for lambda_min  and lambda_max
            y_tilde = SolveBP(Cnw,b,Nbasis,100000,lambda,1e-11);
            y_hat = Wn*y_tilde;%De-normalize coefficients
            Xi = Ww*y_hat;%Unweight coefficients
        
            %Compute l2-norm of residual and l1-norm of solution
            tr_errs(s) = norm(C*Xi - b);
            sol_errs(s) = norm(Xi-Xitrue)/norm(Xitrue);
            sol_l1_norms(s) = norm(y_tilde,1);
        
            xis(s) = log(tr_errs(s));
            etas(s) = log(sol_l1_norms(s));
        
            end

        lambdas(3) = exp((log(lambdas(2)) + pGS*log(lambdas(1)))/(1+pGS));%Store lambda 2
        lambdas(4) = exp(log(lambdas(1)) + log(lambdas(2)) - log(lambdas(3)));%Store lambda 3

        %Solve BPDN for intermediate lambdas 2,3
        for s = 3:4 %Regularization parameter
        
            lambda = lambdas(s);
            fprintf(['Current lambda: ',num2str(lambda),'\n'])
        
            %Solve for lambda_min  and lambda_max
            y_tilde = SolveBP(Cnw,b,Nbasis,100000,lambda,1e-11);
            y_hat = Wn*y_tilde;%De-normalize coefficients
            Xi = Ww*y_hat;%De-normalize coefficients
        
            %Compute l2-norm of residual and l1-norm of solution
            tr_errs(s) = norm(C*Xi - b);%/norm(dxData(:,d));
            sol_errs(s) = norm(Xi-Xitrue)/norm(Xitrue);
            sol_l1_norms(s) = norm(y_tilde,1);
        
            xis(s) = log(tr_errs(s));
            etas(s) = log(sol_l1_norms(s));
        
        end
    
        %Normalize between -1 and 1
        Cshxi = 1 - 2/(xis(2) - xis(1))*xis(2);
        Cscxi = 2/(xis(2) - xis(1));
        Csheta = 1 - 2/(etas(1) - etas(2))*etas(1);
        Csceta = 2/(etas(1) - etas(2));
        xishat = Cshxi + Cscxi*xis;
        etashat = Csheta + Csceta*etas;
        
        %Sort points (xi,eta) corresponding to each lambda in ascending order
        %(lmin < l2 < l3 < lmax)
        P = [xishat(1),xishat(2),xishat(3),xishat(4);etashat(1),etashat(2),etashat(3),etashat(4)];
        %Sort points (xi,eta) corresponding to each lambda in ascending order
        [lambdas, indx] = sort(lambdas);
        P = P(:,indx);%Sort points according to values of lambda in ascending order

        plot(axLcurveIter,P(1,:),P(2,:),'Color','b','Marker','o')
        drawnow

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
        
        %Solve for lambda2
        y_tilde = SolveBP(Cnw,b,Nbasis,100000,lambdas(2),1e-11);
        y_hat = Wn*y_tilde;%De-normalize coefficients  
        Xi = Ww*y_hat;%De-normalize coefficients
        
        %Compute l2-norm of residual and l1-norm of solution
        tr_err = norm(C*Xi - b);%/norm(dxData(:,d));
        sol_err = norm(Xi-Xitrue)/norm(Xitrue)
        sol_l1_norm = norm(y_tilde,1);
        
        xi = log(tr_err);
        eta = log(sol_l1_norm);
        
        xihat = Cshxi + Cscxi*xi;
        etahat = Csheta + Csceta*eta;
        
        P(:,2) = [xihat;etahat];
        
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
        
        %Solve for lambda2
        y_tilde = SolveBP(Cnw,b,Nbasis,100000,lambdas(2),1e-11);
        y_hat = Wn*y_tilde;%De-normalize coefficients  
        Xi = Ww*y_hat;%De-normalize coefficients
        
        %Compute l2-norm of residual and l1-norm of solution
        tr_err = norm(C*Xi - b);%/norm(dxData(:,d));
        lambdas(2)
        sol_err = norm(Xi-Xitrue)/norm(Xitrue)
        sol_l1_norm = norm(y_tilde,1);
 
        xi = log(tr_err);
        eta = log(sol_l1_norm);
        
        xihat = Cshxi + Cscxi*xi;
        etahat = Csheta + Csceta*eta;
        
        P(:,2) = [xihat;etahat];

        plot(axLcurveIter,xihat,etahat,'Color','k','Marker','o','MarkerSize',5,'LineStyle','none')
        drawnow
        plot(axErrorIter,lambdas(2),sol_err,'Color','k','Marker','o','MarkerSize',5,'LineStyle','none')
        drawnow
    
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
        
        %Solve for lambda3
        y_tilde = SolveBP(Cnw,b,Nbasis,100000,lambdas(3),1e-11);
        y_hat = Wn*y_tilde;%De-normalize coefficients 
        Xi = Ww*y_hat;%De-normalize coefficients
        
        %Compute l2-norm of residual and l1-norm of solution
        tr_err = norm(C*Xi - b);%/norm(dxData(:,d));
        lambdas(3)
        sol_err = norm(Xi-Xitrue)/norm(Xitrue)
        sol_l1_norm = norm(y_tilde,1);
        
        xi = log(tr_err); 
        eta = log(sol_l1_norm);
        
        xihat = Cshxi + Cscxi*xi;
        etahat = Csheta + Csceta*eta;
        
        P(:,3) = [xihat;etahat];
    
        plot(axLcurveIter,xihat,etahat,'Color','k','Marker','o','MarkerSize',5,'LineStyle','none')
        drawnow
        plot(axErrorIter,lambdas(2),sol_err,'Color','k','Marker','o','MarkerSize',5,'LineStyle','none')
        drawnow
        
    end
    
    %Compute relative gap
    
    gap = (lambdas(4) - lambdas(1))/lambdas(4);
    lambda_itr2(itr2) = lambdaC;
    itr2 = itr2 + 1;    
    
        end
    pause
    close(figLcurveIter)
    close(figErrorIter)
    
    %% Pick the corner
    lambda_corner = lambdaC;
    lambda_itr(itr) = lambda_corner;
    fprintf(['Optimal lambda: ',num2str(lambda_corner)],'\n\n')

        
        % Solve L1 minimization with optimal sigma
        y_tilde = SolveBP(Cnw,b,Nbasis,100000,lambda_corner,1e-11);
        Xi = Ww*Wn*y_tilde%De-normalize coefficients
        coefferr(itr+1) = norm(Xi - Xitrue)/norm(Xitrue)
        fprintf(['Coefficient error',' = ',num2str(coefferr(itr)),'\n'])
        lambda_corner
        
        sol_l1_normCorner(j,d,itr+1) = norm(y_tilde,1);
        tr_errsCorner(j,d,itr+1) = norm(C*Xi - b);
%         pause
%         close all
    coefferr_noise(j,d,itr+1) = coefferr(itr+1)
    end
    
    Xi_noise(j,d,:) = Xi;
    pause
    close
    
end%End loop for degrees of freedom


end%End loop for noise levels
        

save('Lorenz63_Data_Plots_200','Xi_noise','coefferr_noise','tr_errLc_noise','sol_errLc_noise','sol_l1_normLc_noise','sol_l1_normCorner','tr_errsCorner','sigmas')








 