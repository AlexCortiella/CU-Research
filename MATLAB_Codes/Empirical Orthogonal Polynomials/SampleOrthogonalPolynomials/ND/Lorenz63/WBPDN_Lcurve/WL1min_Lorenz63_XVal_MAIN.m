%% SPARSE IDENTIFICATION OF NONLINEAR DYNAMICAL SYSTEMS

clear all, close all, clc
addpath('./utils','spgl1-1.9');

%% Load data
load('Lorenz63DataStruc200');

% Order of expansion
p = 3;

%Gather data
Nsamp_total = 200;
num_itr = 1;
Ndofs = size(xDataT,2);

%True data
xDataT = xDataT(1:Nsamp_total,:);
dxDataT = dxDataT(1:Nsamp_total,:);
tDataT = tData(1:Nsamp_total,:);
dtDataT = mean(tDataT(2:end) - tDataT(1:end-1));

%Noisy data
xData = xDataN(1:Nsamp_total,:);
dxData = dxDataN(1:Nsamp_total,:);

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
sigmas = logspace(-6,2,200);
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
for d = 1:3 %Degree of freedom/dimension loop
    fprintf(['Degree of freedom: ', num2str(d),'\n'])
    %% Loop over the regularization parameter
            for s = 1:Nsigmas %Regularization parameter
                
            sigma = sigmas(s);
            fprintf(['Current sigma: ',num2str(sigma),'\n'])
            
                for k = 1:K %Cross validation loop
                fprintf(['Current fold: ',num2str(k),'\n'])
                
                %% Generate validation set
                valid = (cvindx == k); %Validation set
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
                y_tilde = SolveBP(CTr,dxDataTr(:,d),Nbasis,100000,sigma,1e-9);
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
%             sigma_opt = sqrt((Nva + Ntr)/Ntr)*sigmas(indx);
            sigma_opt = sigmas(indx);
            
            semilogx(sigma_opt,mean_errZ(indx),'k.','MarkerSize',20);
            save('WL1min_Lorenz63_Xval_0thIteration_2Dof','sigmas','mean_errZ','sigma_opt','min_err');
            fprintf(['Optimal sigma: ',num2str(sigma_opt),' with mean residual: ',num2str(min_err)])
            
            %% Solve L1 minimization with optimal sigma
                %Solve L1 min
                y_tilde = SolveBP(CN,dxData(:,d),Nbasis,100000,sigma,1e-9);
                Xi(:,d) = diag(1./Ccol_norm)*y_tilde;
                coefferr(1,d) = norm(Xi(:,d) - XiT(:,d))/norm(XiT(:,d)); 
                fprintf(['Coefficient error dof = ',num2str(d),':',num2str(coefferr(1,d)),'\n'])
                %% End of initial condition
    
    fprintf('Zeroth iteration: \n')
    Xi
    pause
    close
    fprintf('Start iterative loop... \n\n')
    % Start iteration over data
                for itr = 1:num_itr
                    fprintf(['Iteration: ',num2str(itr),'\n'])
%                     Nsamp = floor(itr * Nsamp_total/num_itr);
                    Nsamp = Nsamp_total;

                    cvindxi = crossvalind('Kfold',Nsamp,K);
                %% Loop over the regularization and penalty parameters
%                     for p = 1:Nparams %Regularization and penalty parameters
                    for p = 1:Nqs  
                        
                        q = qs(p);
                        
                        for s = 1:Nsigmas;
                            
                        sigma = sigmas(s);
                        
                        fprintf(['Current (sigma,q) pair: ','(',num2str(sigma),',',num2str(q),')','\n\n'])  
                        eps_w = 0.0001;
                        
                        for k = 1:K %Cross validation loop
                            fprintf(['Current fold: ',num2str(k),'\n'])
                            %% Generate validation set
                            valid = (cvindxi == k); %Validation set
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
                            
                            %Apply weighting 
                            Wd = 1./(abs(Xi(:,d)).^q+eps_w); 
                            CTrd = CTr * diag(1./Wd);
                            %Solve training set
                            y_tilde = SolveBP(CTrd,dxDataTr(:,d),Nbasis,100000,sigma,1e-9);
                            XiTr = diag(1./col_norm)*(diag(1./Wd)*y_tilde);
                            %Validate data
                            val_errs(k) = norm(CVal*XiTr - dxDataVal(:,d))/norm(dxDataVal(:,d));
                            sol_l1norm(k) = norm(XiTr,1);
                            
                            fprintf(['Validation residual in fold ',num2str(k),': ',num2str(val_errs(k)),'\n\n'])
            
                        end
            
                        mean_err(p,s) = mean(val_errs);
%                         l1_norm(p) = mean(sol_l1norm);


                        fprintf(['Mean residual of pair ',num2str(p),' : ',num2str(mean_err(p)),'\n\n\n'])
                        pointsize = 20;
%                         scatter(sigma, q, pointsize, mean_err(p),'filled') 
                        plot3(sigma, q, mean_err(p,s),'r*-') 
                        xlabel('sigma');
                        ylabel('q')
                        grid('on')
%                         colorbar
                        hold on;
                        drawnow;
                        end
                        
                        pause                    
                    end
                    pause
                    close
                    %Pick the sigma/q pair corresponding to the smallest mean error
                    minres = min(min(mean_err));
                    [rind,cind] = find(mean_err == minres);
%                     [min_err,indx] = min(mean_err);
%                     [min_err,indx] = min(l1_norm);

%                     sigma_opt = sqrt((Nva + Ntr)/Ntr)*params(1,indx);
                    sigma_opt = sqrt((Nva + Ntr)/Ntr)*sigmas(cind);
%                     q_opt = sqrt((Nva + Ntr)/Ntr)*params(2,indx);
                    q_opt = qs(rind);
                    
                    fprintf(['Optimal (sigma/q) pair: (',num2str(sigma_opt),' ,',num2str(q_opt),') with mean residual: ',num2str(minres),'\n\n'])

                    %% Solve L1 minimization with optimal sigma/q pair
                    Cit = C(1:Nsamp,:);
                    col_norm = zeros(1,Nbasis);
                    for j = 1:Nbasis
                        col_norm(j) = norm(Cit(:,j));
                    end
                    Cit = Cit * diag(1./col_norm);
                    
                    %Weighted L1 minimization
                    Wd = 1./(abs(Xi(:,d)).^q_opt+eps_w); % The weight matrix
                    Citd = Cit * diag(1./Wd);
                    y_tilde = SolveBP(Citd,dxData(1:Nsamp,d),Nbasis,100000,sigma,1e-9);
                    Xi(:,d) = diag(1./col_norm)*(diag(1./Wd)*y_tilde)
        
                    coefferr(itr+1,d) = norm(Xi(:,d) - XiT(:,d))/norm(XiT(:,d));
                    fprintf(['Coefficient error dof = ',num2str(d),' at iteration ',num2str(itr),' : ',num2str(coefferr(itr+1,d)),'\n\n\n'])
                    pause
                end
                close
    
end 





    
    
    
 