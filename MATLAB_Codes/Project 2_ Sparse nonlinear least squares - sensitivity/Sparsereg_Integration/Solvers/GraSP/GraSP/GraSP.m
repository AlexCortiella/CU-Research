function xhat = GraSP(funObj,s,n,options)
% Gradient Support Pursuit Algorithm
% INPUTS:
% funObj: The cost function handle. It has the form funObj(x,I), where I determines the coordinates that the cost function is restricted to,
%        and x is the point at which the cost function is evaluated. To get the unrestricted function I must be set to 1:n with n being the
%        dimensionality of the domain of the cost function.
% s: An integer determining the desired sparsity level
% n: An integer indicating the dimensionality of the argument of the cost function
% options: A structure variable determining various options of the algorithm through the following fields:
%           .HvFunc [default = []] : A function handle for Hessian-vector multiplication. The input arguments are supposed to be passed as HvFunc(v,x,I),
%                                   where I determines the coordinates that the cost function is restricted to, x is the point at which the Hessian of the
%                                   restricted function is evaluated, and v is the vector to be multiplied by the restriced Hessian.         
%           .Method [default = 'F']: Takes values 'F', 'G', and 'H' corresponding to the following variants for the inner optimization of GraSP:
%                 'F': the optimization is performed *fully*
%                 'G': only a *gradient descent* step is performed
%                 'H': only a *newton* step is performed
%           .mu [default = 0]: A real number whose absolute value determines either the L2-regularization coefficient (if mu<=0) or the radius of sphere the iterates will be restricted to (if mu>0).
%           .eta [default = 1]: A positive real number corresponding to the step size for Method = 'G' or 'H'
%           .maxIter [default = 100]: An integer indicating the maximum number of the GraSP iterations to be performed
%           .tolF [default = 1e-6]: GraSP halts if the decrease in the value of the cost function is less than the positive real number tolF
%           .tolG [default = 1e-3]: GraSP halts if the 3s-largest coordinates of the gradient of the cost have an l2-norm less than tolG
  
    flds = {'HvFunc','Method','mu','eta','tolF','tolG','maxIter','refit'}; 
    vals = {[],'F', 0, 1, 1e-6, 1e-3,100,false}; % Default values of the fields in 'options'
    if exist('options','var')  
        for k = 1:numel(flds)
            if isfield(options,flds{k});
                vals{k} = options.(flds{k});
            end
        end
    end
    [HvMult, Method, mu, eta, tolF, tolG, maxIter, refit] = deal(vals{:}); % Set the options of GraSP  
      
    mup = -mu.*(mu<0);
    
    % Configure the 3rd Party Algorithm for the Full Mode
    opts.optTol = 1e-6;
    opts.progTol = 1e-10;    
    if mu <= 0
        opts.Display = 'off';
        if isempty(HvMult)
            opts.Method = 'newton';        
        else
            opts.Method = 'newton0';                        
        end        
        % opts.cgSolve = 1;
        % options.LS_init = 2;
        % opts.HessianModify = 2;
        % opts.useMex = 1;
        % opts.HessianIter = 5;
        
    else        
        opts.verbose = 0;
    end      

    % Initialize the Estimate
    xhat = zeros(n,1);
    Shat = [];
          
    % Initialize the Record of The Achieved Cost Values
    fhistory = NaN(1,maxIter);
         
    % Main Iterations of the GraSP Algorithm
    for j = 1:maxIter
        % Find Gradient
        [fc, z] = augmentedCost(xhat);
%         fprintf('iter# = %d, f = %f, ||g|| = %f\n',j,fc,norm(z));      
                
        % Identify Dominant Gradient Direction
        [~, sSupp] = sort(abs(z),'descend');    
        Omega = sSupp(1:min(numel(z),2*s));
                
        % Check if Halting Condition Holds   
        HaltCond =  (abs(fhistory(max(j-1,1)) - fc - tolF/2) < tolF/2) || (norm(z(sSupp(1:min(numel(z),3*s)))) < tolG) || any(fhistory == fc);
        if HaltCond        
            break;
        end
        
        % Update the Record
        fhistory(j) = fc;
        
        % Merge Supports
        T = union(Omega,Shat);

        % Signal Estimation
        b = zeros(n,1);           
        switch Method
            case 'F'                       
                if ~isempty(HvMult)
                    if mu < 0
                        opts.HvFunc = @(v,xx)(HvMult(v,xx,T)+0.5*mup*(xx'*v)*xx);                        
                    else
                        opts.HvFunc = @(v,xx)HvMult(v,xx,T);                                              
                    end
                end
                if mu <= 0 
                    b(T) = minFunc(@augmentedCost,zeros(numel(T),1),opts);
                else
                    b(T) = minConf_PQN(@(xx)funObj(xx,T),zeros(numel(T),1),@ProjectOnSphere,opts);
                end
            case 'G'
                [~, df] = augmentedCost(xhat(T));
                b(T) = xhat(T) - eta*df;
            case 'H'
                [~, df, d2f] = augmentedCost(xhat(T));
                b(T) = xhat(T) - eta*d2f\df;
        end
        
        % Prune Estimate
        xhat = zeros(n,1);       
        [~,Shat] = sort(abs(b),'descend');
        Shat = Shat(1:s);       
        if refit
            optsRefit = opts;
            optsRefit.HvFunc = @(v,xx)HvMult(v,xx,Shat);
            if mu <= 0
                xhat(Shat) = minFunc(@(xx)funObj(xx,Shat),zeros(s,1),optsRefit);
            else
                xhat(Shat) = minConf_PQN(@(xx)funObj(xx,Shat),zeros(s,1),@ProjectOnSphere ,optsRefit);
            end
        else
            xhat(Shat) = b(Shat);              
        end
    end
     
    function [ff gg HH] = augmentedCost(x)
    % Augment the Cost Function by 0.5*mup*||x||_2^2.

        if ~exist('T','var') || numel(x)>numel(T)
            I = 1:numel(x);
        else
            I = T;
        end
        % Augment, the Cost Function, Its Gradient, and Its Hessian  
        if nargout == 1
            ff = funObj(x,I);
        elseif nargout == 2
            [ff gg] = funObj(x,I);
            gg = gg + mup*x;
        else
            [ff gg HH] = funObj(x,I);        
            gg = gg + mup*x;
            HH = HH + mup*eye(numel(I));
        end    
        ff = ff + 0.5*mup*(x'*x);
    end

    function P = ProjectOnSphere(V)
    % Projection onto a sphere of radius mu
        P = V;    
        if mu > 0
            R = sqrt(sum(V.^2,1));
            id = R > mu;
            if any(id)
                P(id) = mu * bsxfun(@times,V(id,:),1./R(id));
            end
        end
    end
end