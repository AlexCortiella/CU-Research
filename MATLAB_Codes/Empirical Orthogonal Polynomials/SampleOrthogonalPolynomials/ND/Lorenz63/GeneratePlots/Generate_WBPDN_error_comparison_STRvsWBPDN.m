clc;
close all;
clear all;

filename = 'Lorenz63_comparison_solerror_noise';

load('STRidgeResultsMean');
load('Lorenz63_Data_Plots_400');

coefferr_noise_WBPDN = coefferr_noise;
coefferr_noise_STR = coefferr_noiseSTR_mean;

plot_coefferr_noise_3dof_comparison(sigmas, [squeeze(coefferr_noise_WBPDN(:,:,end)), squeeze(coefferr_noise_STR(:,:,end))])
print(filename,'-depsc')
