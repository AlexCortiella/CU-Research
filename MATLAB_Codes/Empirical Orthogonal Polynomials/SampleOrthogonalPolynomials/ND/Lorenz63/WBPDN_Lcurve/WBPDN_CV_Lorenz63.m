%% SPARSE IDENTIFICATION OF NONLINEAR DYNAMICAL SYSTEMS

clear all, close all, clc
addpath('./Solvers','./utils');
rng(100)
%% Load data
load('Lorenz63DataStruc200');
% Order of expansion
p = 3;
% p = 4;
Nsamp_total = 200;


%Noise levels
Nsigmas = length(Lorenz63Data);

% True coefficient vector in monomial basis
Nb = nchoosek(p+3,3);
XiT = zeros(Nb,3);
sig = Lorenz63Data(1).paramsValue(1);
rho = Lorenz63Data(1).paramsValue(2);
beta = Lorenz63Data(1).paramsValue(3);

%% START CROSS VALIDATION LOOP

K = 5; %number of folds
cvindx = crossvalind('Kfold',Nsamp_total,K);%Create indices for cross-validation
lambdas = logspace(log10(eps),2,1000);
Nlambdas = length(lambdas);%Number of sigmas
%Define penalization parameter (exponents of the weighting matrix) 
%Number of parameters

val_errs = zeros(1,K);
sol_l1norm = zeros(1,K);
mean_err = zeros(1,Nlambdas);

XiT(2,1) = -sig;
XiT(2,2) = rho;
XiT(3,1) = sig;
XiT(3,2) = -1;
XiT(4,3) = -beta;
XiT(6,3) = 1;
XiT(8,2) = -1;

noise = 4;
%Gather data

%True data
xDataT = Lorenz63Data(noise).xtrue(1:Nsamp_total,:);
dxDataT = Lorenz63Data(noise).dxtrue(1:Nsamp_total,:);
tDataT = Lorenz63Data(noise).time(1:Nsamp_total,:);
dtDataT = mean(tDataT(2:end) - tDataT(1:end-1));

%Noisy data
xData = Lorenz63Data(noise).xnoisy(1:Nsamp_total,:);
dxData = Lorenz63Data(noise).dxnoisy(1:Nsamp_total,:);


% Setup the basis
Ndofs = size(xDataT,2);
index_pc = nD_polynomial_array(Ndofs,p);
Nbasis = size(index_pc,1);

%Allocate identified coefficients
Xid = zeros(Nbasis,Ndofs);
coefferr = zeros(Ndofs,1);% NOTE: the first row represents the solution error at 0th iteration

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
    
   %% Loop over the regularization parameter
            for s = 1:Nlambdas %Regularization parameter
                
            lambda = lambdas(s);
            fprintf(['Current lambda: ',num2str(lambda),'\n'])
            
                for k = 1:K %Cross validation loop
                fprintf(['Current fold: ',num2str(k),'\n'])
                
                %% Generate validation set
                valid = (cvindx == k); %Validation set
                Nva = sum(valid);
                bVal = b(valid);
                
                % Form the measurement matrix
                CVal = C(valid,:); 
            
                %% Generate training set
                train = ~valid;%Training set
                Ntr = sum(train);
                bTr = b(train);
                % Form the measurement matrix
                CTr = C(train,:);                
                % Normalize columns
                col_norm = sqrt(sum(CTr.*CTr,1));
                WnTr = diag(1./col_norm);
                CTr = CTr * WnTr;
                
                %Solve BPDN for the training set
                y_tilde = SolveBP(CTr,bTr,Nbasis,100000,lambda,1e-11);
                XiTr = WnTr*y_tilde;%De-normalize coefficients
                %Validate data
                val_errs(k) = norm(CVal*XiTr - bVal)/norm(bVal);
                sol_l1norm(k) = norm(XiTr,1);

                fprintf(['Validation residual in fold ',num2str(k),': ',num2str(val_errs(k)),'\n\n'])
                end
                mean_err(s) = mean(val_errs);

                fprintf(['Mean residual for lambda s = ',num2str(lambda),': ', num2str(mean_err(s)),'\n'])
            end
            
            %Plot mean error vs sigmas
            figure(1)
            semilogx(lambdas,mean_err,'r.-');
            ylabel('Mean relative residual')
            xlabel('\lambda')
            grid('on')
            hold on
            
            %Pick the sigma corresponding to the smallest mean error
            [min_err,indx] = min(mean_err);
            lambda_opt(d) = sqrt((Nva + Ntr)/Ntr)*lambdas(indx);
            
            semilogx(lambda_opt(d),mean_err(indx),'k.','MarkerSize',20);
            fprintf(['Optimal sigma: ',num2str(lambda_opt(d)),' with mean residual: ',num2str(min_err)])
            
            %% Solve L1 minimization with optimal sigma
                %Solve L1 min
                %Solve BPDN for the training set
                y_tilde = SolveBP(Cn,b,Nbasis,100000,lambda_opt(d),1e-11);
                Xi = Wn*y_tilde;%De-normalize coefficients
                Xi_CV(:,d) = Xi;
                resCV(d) = norm(C*Xi - b);
                regCV(d) = norm(y_tilde,1);
                coefferr(1,d) = norm(Xi - XiT(:,d))/norm(XiT(:,d)); 
                fprintf(['Coefficient error dof = ',num2str(d),':',num2str(coefferr(1,d)),'\n'])
                %% End of initial condition
                
end
    
    fprintf('Zeroth iteration: \n')
    Xid
    
    save('Lorenz63_CVData','resCV','regCV','lambda_opt','Xi_CV')



