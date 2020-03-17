clc;
close all;
clear all;

filename = 'Lorenz63_WBPDN_solerror_iter_noise0d01_200';


load('Lorenz63_Data_Plots_200');
noise = 4;

plot_coefferr_iter_3dof(0:size(coefferr_noise,3)-1, (squeeze(coefferr_noise(noise,:,:)))')
print(filename,'-depsc')