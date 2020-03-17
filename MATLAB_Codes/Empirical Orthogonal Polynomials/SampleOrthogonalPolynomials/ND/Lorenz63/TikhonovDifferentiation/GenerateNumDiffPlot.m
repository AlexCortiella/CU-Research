clc;
close all;
clear all;
filename = 'Lorenz63_NumDiffPlots';
load('Lorenz63DataStruc');

sigmas = Lorenz63Data(1).sigma;
Nsigmas = length(sigmas);
Nstates = 3;

dxerror = size(Nsigmas,Nstates);


for j = 1:Nsigmas
    
    for d = 1:Nstates
        
        dxerror(j,d) = norm(Lorenz63Data(j).dxnoisy(:,d) - Lorenz63Data(j).dxtrue(:,d))/norm(Lorenz63Data(j).dxtrue(:,d));
        
    end
    
end

% plotNumericalDifferentiation(sigmas, dxerror)
plotNumDiff(sigmas, dxerror)

print(filename,'-depsc')