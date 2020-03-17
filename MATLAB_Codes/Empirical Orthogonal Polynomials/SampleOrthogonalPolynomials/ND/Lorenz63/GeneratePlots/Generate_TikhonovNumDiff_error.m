clc;
close all;
clear all;
filename = 'Lorenz63_TikhonovNumDiff_dxerror_200samp';
load('Lorenz63TikhonovNumDiff200');

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
plot_NumDiff_error_3dof(sigmas, dxerror)

print(filename,'-depsc')