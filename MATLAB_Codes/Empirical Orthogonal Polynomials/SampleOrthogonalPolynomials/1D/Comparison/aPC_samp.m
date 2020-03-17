%% aPC Matlab Toolbox
% Data-driven Arbitrary Polynomial Chaos Expansion
% Author: Sergey Oladyshkin
% Stuttgart Research Centre for Simulation Technology
% Department of Stochastic Simulation and Safety Research for Hydrosystems,
% Institute for Modelling Hydraulic and Environmental Systems
% University of Stuttgart, Pfaffenwaldring 5a, 70569 Stuttgart
% E-mail: Sergey.Oladyshkin@iws.uni-stuttgart.de
% Phone: +49-711-685-60116
% Fax: +49-711-685-51073
% http://www.iws.uni-stuttgart.de

% The current aPC Matlab Toolbox is using definition of aPC that is presented in the following manuscripts: 
% Oladyshkin S. and Nowak W. Data-driven uncertainty quantification using the arbitrary polynomial chaos expansion. Reliability Engineering & System Safety, Elsevier, V. 106, P. 179�190, 2012.
% Oladyshkin S. and Nowak W. Incomplete statistical information limits the utility of high-order polynomial chaos expansions. Reliability Engineering & System Safety, 169, 137-148, 2018.

%% Construction of Data-driven Arbitrary Orthonormal Polynomial Basis
function [OrthonormalBasis, MonicBasis] = aPC_samp(Data, Degree)
% Input:
% Data - raw data array
% Degree - degree of the orthonormal polynomial basis
% Output:
% OrthonormalBasis - orthonormal polynomial basis 

%% Initialization
d=Degree; %Degree of polynomial expansion
% dd=d+1; %Degree of polynomial defenition
dd=d; %Degree of polynomial defenition

L_norm=1; % L-norm for polynomial normalization
NumberOfDataPoints=length(Data);

%% Forward linear transformation
MeanOfData=mean(Data);
VarOfData=var(Data);
Data=Data/MeanOfData;

%% Raw Moments of the Input Data
for i=0:(2*dd+1)
    m(i+1)=sum(Data.^i)/length(Data); 
end

%% Polynomial up to degree dd
for degree=0:dd; %Note that the moment matrix is generated for each degree (the moment matrix scales with the degree)
    %% Generate moment matrix
    %Definition of Moments Matrix Mm (Bulid the moment matrix)
    for i=0:degree;
        for j=0:degree;                    
            if (i<degree) 
                Mm(i+1,j+1)=m(i+j+1); 
            end
            %Build the last row (all zeros except last entry)
            if (i==degree) && (j<degree)
                Mm(i+1,j+1)=0;
            end
            if (i==degree) && (j==degree)
                Mm(i+1,j+1)=1;
            end        
        end
        %Matrix Normalization(normalize the moment matrix with the maximum absolute value of the columns for row i) 
        Mm(i+1,:)=Mm(i+1,:)/max(abs(Mm(i+1,:))); 
    end
    %% Generate RHS
    %Definition of ortogonality conditions Vc (generate RHS, column vector
    %containing all zeros except last entry)
    for i=0:degree;
        if (i<degree) 
             Vc(i+1)=0; 
        end            
        if (i==degree)
            Vc(i+1)=1;
        end
    end
    
    %% Solve linear system M*p = e for polynomial coefficients p (Note that these coefficients correspond to monic polynomials since the last entry of Vc is 1) 
    %Coefficients of Non-Normal Orthogonal Polynomial: Vp
    inv_Mm=pinv(Mm);
    Vp=Mm\Vc';
    cond(Mm)
    res = norm(Mm*Vp - Vc');
    %% Construct coefficient matrix
    PolyCoeff_NonNorm(degree+1,1:degree+1)=Vp'; 
%     degree
%     PolyCoeff_NonNorm
%     pause

    %CM =[p_0^(0)   0 0 0 0 0 0 ... 0 0 0 0 0;
    %     p_0^(1) p_1^(1) 0 0 0 ... 0 0 0 0 0;
    %     p_0^(2) p_1^(2) p_2^(2)...  0 0 0 0;  
    %     ...    ...   ...   ...             ]
       
    fprintf('=> aPC Toolbox: Inaccuracy for polynomial basis of degree %1i is %5f pourcents',degree, 100*abs(sum(abs(Mm*PolyCoeff_NonNorm(degree+1,1:degree+1)'))-sum(abs(Vc)))); 
    if 100*abs(sum(abs(Mm*PolyCoeff_NonNorm(degree+1,1:degree+1)'))-sum(abs(Vc)))>0.5
        fprintf('\n=> Warning: Computational error of the linear solver is too high.');           
    end
    fprintf('\n');
    fprintf('=> Norm of the residual res = Mm*Vp - Vc for degree %1i is %5d',degree,res); 
    fprintf('\n');
    %% Normalize coefficients to obtain an orthonormal basis
    %Normalization of polynomial coefficients
    P_norm=0;
    for i=1:NumberOfDataPoints;        
        Poly=0;
        %Compute the polynomial for data point i (We want to compute the
        %inner product of each polynomial with itself with respect to the discrete sample measure
        %<pk,pk> = 1/(Nsamples)*sum_isamples(pk(isample)^2). This will give
        %the squared normalization constant
        for k=0:degree;
            Poly=Poly+PolyCoeff_NonNorm(degree+1,k+1)*Data(i)^k;     
        end
        P_norm=P_norm+Poly^2/NumberOfDataPoints;%Sum over all sample points        
    end
    %Take the square root of the squared normalization constant
    P_norm=sqrt(P_norm);
    %Normalize coefficients for each polynomial
    for k=0:degree;
        Polynomial(degree+1,k+1)=PolyCoeff_NonNorm(degree+1,k+1)/P_norm;
    end
end

%% Backward linear transformation to the data space (because we used a normalized moment matrix with respec to the sample mean)
Data=Data*MeanOfData;
for k=1:length(Polynomial);
    Polynomial(:,k)=Polynomial(:,k)/MeanOfData^(k-1);
    PolyCoeff_NonNorm(:,k) = PolyCoeff_NonNorm(:,k)/MeanOfData^(k-1);
end

%% Data-driven Arbitrary Orthonormal Polynomial Basis
OrthonormalBasis=Polynomial;
MonicBasis=PolyCoeff_NonNorm;
