clc;
close all;
clear all;

filename = 'Lorenz63_WBPDN_solerror_noise_200';


load('Lorenz63_Data_Plots_400');

plot_coefferr_noise_3dof(sigmas, squeeze(coefferr_noise(:,:,end)))
print(filename,'-depsc')
