%% SPARSE IDENTIFICATION OF NONLINEAR DYNAMICAL SYSTEMS

clear all, close all, clc
addpath('./utils','Solvers');
rng(100)
%% Load data
load('Lorenz63DataStruc200');

% Order of expansion
p = 3;

%Gather data
Nsamp_total = 200;
num_itr = 1;
Ndofs = 3;
noise = 4;

%True data
xDataT = Lorenz63Data(noise).xtrue(1:Nsamp_total,:);
dxDataT = Lorenz63Data(noise).dxtrue(1:Nsamp_total,:);
tDataT = Lorenz63Data(noise).time(1:Nsamp_total);
dtDataT = mean(tDataT(2:end) - tDataT(1:end-1));

%Noisy data
xData = Lorenz63Data(noise).xnoisy(1:Nsamp_total,:);
dxData = Lorenz63Data(noise).dxnoisy(1:Nsamp_total,:);
% Setup the basis
index_pc = nD_polynomial_array(Ndofs,p);
Nbasis = size(index_pc,1);


% True coefficient vector in monomial basis
XiT = zeros(Nbasis,Ndofs);
XiT(2,1) = -10;
XiT(2,2) = 28;
XiT(3,1) = 10;
XiT(3,2) = -1;
XiT(4,3) = -8/3;
XiT(6,3) = 1;
XiT(8,2) = -1;


%Allocate identified coefficients
Xi = zeros(Nbasis,Ndofs);


%% START CROSS VALIDATION LOOP

K = 5; %number of folds
cvindx = crossvalind('Kfold',Nsamp_total,K);%Create indices for cross-validation

%Define sigma and p ranges to loop over
%Define sigmas (min |x|_1 s.t. |Ax - b|_2 < sigma)
% sigmas = [10.^(linspace(-1,-4,Nsigmas))];
% sigmas = [0.1,0.05,0.01,0.005,0.001,0.0005,0.0001];
% sigmas = [1e2,5e1,1e1,5e0,1e0,5e-1,2e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4];
sigmas = logspace(log10(eps),2,1000);
Nsigmas = length(sigmas);%Number of sigmas
%Define penalization parameter (exponents of the weighting matrix) 
%Number of parameters
qs = [2];
Nqs = length(qs);

val_errs = zeros(1,K);
sol_l1norm = zeros(1,K);
mean_errZ = zeros(1,Nsigmas);

params = combvec(sigmas,qs);%Generate an array with all combinations of sigma and p parameters
Nparams = length(params);
mean_err = zeros(Nqs,Nsigmas);

coefferr = zeros(num_itr+1,Ndofs);

%Build dictionary for the entire data set
C = zeros(Nsamp_total,Nbasis);
for isim = 1:Nsamp_total
    crow = piset_monomial(xData(isim,:),index_pc);
    C(isim,:) = crow(1:Nbasis);
end
 %Normalize columns
Ccol_norm = sqrt(sum(C.*C,1));

CN = C * diag(1./Ccol_norm);

%% Zeroth iteration (no weighting)
for d = 3:3 %Degree of freedom/dimension loop
    fprintf(['Degree of freedom: ', num2str(d),'\n'])
    %% Loop over the regularization parameter
            for s = 1:Nsigmas %Regularization parameter
                
            sigma = sigmas(s);
            fprintf(['Current sigma: ',num2str(sigma),'\n'])
            
                for k = 1:K %Cross validation loop
                fprintf(['Current fold: ',num2str(k),'\n'])
                
                %% Generate validation set
                valid = (cvindx == k); %Validation set
                find(valid)
                Nva = sum(valid);
                xDataVal = xData(valid,:);
                dxDataVal = dxData(valid,:);
               
                % Form the measurement matrix
                CVal = C(valid,:); 
            
                %% Generate training set
                train = ~valid;%Training set
                Ntr = sum(train);
                xDataTr = xData(train,:);
                dxDataTr = dxData(train,:);
                % Form the measurement matrix
                CTr = C(train,:);
                % Normalize columns
                col_norm = sqrt(sum(CTr.*CTr,1));
                CTr = CTr * diag(1./col_norm);
                %Solve training set
%                 [y_tilde,res,gap,info] = spg_bpdn(CTr,dxDataTr(:,d),sigma*norm(dxDataTr(:,d)),opts);
                y_tilde = SolveBP(CTr,dxDataTr(:,d),Nbasis,100000,sigma,1e-11);
                XiTr = diag(1./col_norm)*y_tilde%De-normalize coefficients
                %Validate data
                val_errs(k) = norm(CVal*XiTr - dxDataVal(:,d))/norm(dxDataVal(:,d));
                sol_l1norm(k) = norm(XiTr,1);

                fprintf(['Validation residual in fold ',num2str(k),': ',num2str(val_errs(k)),'\n\n'])
                end

                mean_errZ(s) = mean(val_errs);
%                 l1_normZ(s) = mean(sol_l1norm);

                fprintf(['Mean residual for sigma s = ',num2str(sigma),': ', num2str(mean_errZ(s)),'\n'])
            end
            
            %Plot mean error vs sigmas
            figure(1)
            semilogx(sigmas,mean_errZ,'r.-');
            ylabel('Mean relative residual')
            xlabel('sigma')
            grid('on')
            hold on
            
            %Pick the sigma corresponding to the smallest mean error
            [min_err,indx] = min(mean_errZ);
%             [min_err,indx] = min(l1_normZ);
            sigma_opt(d) = sqrt((Nva + Ntr)/Ntr)*sigmas(indx);
%             sigma_opt = sigmas(indx);
            
            semilogx(sigma_opt,mean_errZ(indx),'k.','MarkerSize',20);
            save('WL1min_Lorenz63_Xval_0thIteration_2Dof','sigmas','mean_errZ','sigma_opt','min_err');
            fprintf(['Optimal sigma: ',num2str(sigma_opt),' with mean residual: ',num2str(min_err)])
            
            %% Solve L1 minimization with optimal sigma
                %Solve L1 min
                y_tilde = SolveBP(CN,dxData(:,d),Nbasis,100000,sigma_opt(d),1e-11);
                Xi(:,d) = diag(1./Ccol_norm)*y_tilde;
                resCV(d) = norm(C*Xi(:,d) - dxData(:,d));
                regCV(d) = norm(y_tilde,1);
                coefferr(1,d) = norm(Xi(:,d) - XiT(:,d))/norm(XiT(:,d)); 
                fprintf(['Coefficient error dof = ',num2str(d),':',num2str(coefferr(1,d)),'\n'])
                %% End of initial condition
    
    fprintf('Zeroth iteration: \n')
    Xi
    pause

    
end 

save('Lorenz63_CVData','resCV','regCV','sigma_opt')



    
    
    
 