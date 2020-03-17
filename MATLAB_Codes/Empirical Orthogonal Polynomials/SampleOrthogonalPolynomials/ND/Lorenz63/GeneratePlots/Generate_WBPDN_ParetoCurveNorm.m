clc;
close all;
clear all;
load('Lorenz63_Data_Plots_200');
load('Lorenz63_CVData');

filename = 'Lorenz63_WBPDN_ParetoCurve_noise0d01_iter0_200';


noise = 4;
indx = zeros(3,1);
minres = zeros(3,1);
minreg = zeros(3,1);
iter = 1;



for d = 1:3
    
xiLc = log(tr_errLc_noise(noise,d,iter,:));
etaLc = log(sol_l1_normLc_noise(noise,d,iter,:));

xiCorner = log(tr_errsCorner(noise,d,iter));
etaCorner = log(sol_l1_normCorner(noise,d,iter));

xiCV = log(resCV(d));
etaCV = log(regCV(d));
        
CshxiLc = 1 - 2/(xiLc(end) - xiLc(1))*xiLc(end);
CscxiLc = 2/(xiLc(end) - xiLc(1));
CshetaLc = 1 - 2/(etaLc(1) - etaLc(end))*etaLc(1);
CscetaLc = 2/(etaLc(1) - etaLc(end));
    
xihatLc(d,:) = CshxiLc + CscxiLc*xiLc;
etahatLc(d,:) = CshetaLc + CscetaLc*etaLc;
xihatCorner(d) = CshxiLc + CscxiLc*xiCorner;
etahatCorner(d) = CshetaLc + CscetaLc*etaCorner;
xihatCV(d) = CshxiLc + CscxiLc*xiCV;
etahatCV(d) = CshetaLc + CscetaLc*etaCV;

[minc,indx] = min(sol_errLc_noise(noise,d,iter,:));
minres(d) = xihatLc(d,indx);
minreg(d) = etahatLc(d,indx);

end

x1 = xihatLc(1,:); x2 = xihatCorner(1);  x3 = minres(1); x10 = xihatCV(1);
y1 = etahatLc(1,:); y2 = etahatCorner(1); y3 = minreg(1); y10 = etahatCV(1);

x4 = xihatLc(2,:); x5 = xihatCorner(2);  x6 = minres(2); x11 = xihatCV(2);
y4 = etahatLc(2,:); y5 = etahatCorner(2); y6 = minreg(2); y11 = etahatCV(2);

x7 = xihatLc(3,:); x8 = xihatCorner(3);  x9 = minres(3); x12 = xihatCV(3);
y7 = etahatLc(3,:); y8 = etahatCorner(3); y9 = minreg(3); y12 = etahatCV(3);





plot_ParetoCurves_3dof_norm(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12)

print(filename,'-depsc')