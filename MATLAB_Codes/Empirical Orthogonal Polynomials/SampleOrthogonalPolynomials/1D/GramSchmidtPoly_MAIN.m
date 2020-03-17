%% Gram-Schmidt polynomial orthogonalization 
clc;
close all;
clear all;

%% MONIC basis
%Legendre basis example

%Define support
a = -1;
b = 1;
support = [-1,1];

%Define weighting function
wfun = @(x) 1;

%Generate data points
M = 836;%Number of data points
x = a:(b-a)/(M-1):b;

%Define polynomial degree
deg = 5;

[Qm,Rm] = gramschmidt_monic(x,deg,wfun,support);
%Check orthogonality
Qm'*Qm

[Qn,Rn] = gramschmidt_normal2(x,deg,wfun,support);
%scale basis accounting for the number of samples and support
Qn = sqrt((b-a)/M)*Qn;
%Check orthonormality
Qn'*Qn

[Qnm,Rnm] = modgramschmidt_normal(x,deg,wfun,support);
%scale basis accounting for the number of samples and support
Qnm = sqrt((b-a)/M)*Qnm;
%Check orthonormality
Qnm'*Qnm



