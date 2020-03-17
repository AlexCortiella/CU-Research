clc;
close all;
clear all;
load('Lorenz63Lcurve');
filename = 'Lorenz63_NumDiffLcurvePlots_sig_1e-5';


noise = 4;
indx = zeros(3,1);
minres = zeros(3,1);
minreg = zeros(3,1);

for d = 1:3
[minc,indx] = min(errordxLcnoise(noise,d,:));
minres(d) = residualLcnoise(noise,d,indx);
minreg(d) = regularizerLcnoise(noise,d,indx);
end

x1 = squeeze(residualLcnoise(noise,1,:)); x2 = resCorner(noise,1);  x3 = minres(1);
y1 = squeeze(regularizerLcnoise(noise,1,:)); y2 = regCorner(noise,1); y3 = minreg(1);

x4 = squeeze(residualLcnoise(noise,2,:)); x5 = resCorner(noise,2);  x6 = minres(2);
y4 = squeeze(regularizerLcnoise(noise,2,:)); y5 = regCorner(noise,2); y6 = minreg(2);

x7 = squeeze(residualLcnoise(noise,3,:)); x8 = resCorner(noise,3);  x9 = minres(3);
y7 = squeeze(regularizerLcnoise(noise,3,:)); y8 = regCorner(noise,3); y9 = minreg(3);

plotNumDiffLcurve(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9)

print(filename,'-depsc')