clc;
close all;
clear all;
load('Lorenz63TikhonovNumDiff200');
filename = 'Lorenz63_TikhonovNumDiff_Lcurve_noise0d01_200samp';


noise = 4;
indx = zeros(3,1);
minres = zeros(3,1);
minreg = zeros(3,1);


for d = 1:3
    
xiLc = log(residualLcnoise(noise,d,:));
etaLc = log(regularizerLcnoise(noise,d,:));

xiCorner = log(resCorner(noise,d));
etaCorner = log(regCorner(noise,d));
        
CshxiLc = 1 - 2/(xiLc(end) - xiLc(1))*xiLc(end);
CscxiLc = 2/(xiLc(end) - xiLc(1));
CshetaLc = 1 - 2/(etaLc(1) - etaLc(end))*etaLc(1);
CscetaLc = 2/(etaLc(1) - etaLc(end));
    
xihatLc(d,:) = CshxiLc + CscxiLc*xiLc;
etahatLc(d,:) = CshetaLc + CscetaLc*etaLc;
xihatCorner(d) = CshxiLc + CscxiLc*xiCorner;
etahatCorner(d) = CshetaLc + CscetaLc*etaCorner;

[minc,indx] = min(errordxLcnoise(noise,d,:));
minres(d) = xihatLc(d,indx);
minreg(d) = etahatLc(d,indx);

end

x1 = xihatLc(1,:); x2 = xihatCorner(1);  x3 = minres(1);
y1 = etahatLc(1,:); y2 = etahatCorner(1); y3 = minreg(1);

x4 = xihatLc(2,:); x5 = xihatCorner(2);  x6 = minres(2);
y4 = etahatLc(2,:); y5 = etahatCorner(2); y6 = minreg(2);

x7 = xihatLc(3,:); x8 = xihatCorner(3);  x9 = minres(3);
y7 = etahatLc(3,:); y8 = etahatCorner(3); y9 = minreg(3);

plot_TikLCurves_3dof_norm(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9)

print(filename,'-depsc')