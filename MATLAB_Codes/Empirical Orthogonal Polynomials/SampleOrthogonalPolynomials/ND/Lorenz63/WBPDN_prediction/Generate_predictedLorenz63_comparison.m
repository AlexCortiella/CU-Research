clc;
close all;
clear all;
load('PredictedLorenz63Data');
filename = 'PredictedLorenz63_200samp';


plot_predictedLorenz63(tT, [xT(:,1),xI(:,1)], [xT(:,2),xI(:,2)], [xT(:,3),xI(:,3)])
print(filename,'-depsc')

