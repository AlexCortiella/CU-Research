% STIELTJES Discretized Stieltjes procedure for arbitrary measures from samples.
%
%    Given the discrete inner product whose nodes are contained 
%    in the first column, and whose weights are contained in the
%    second column, of the nx2 array xw, the call ab=STIELTJES(n,xw) 
%    generates the first n recurrence coefficients ab of the 
%    corresponding discrete orthogonal polynomials. The n alpha-
%    coefficients are stored in the first column, the n beta-
%    coefficients in the second column, of the nx2 array ab.
%
function ab=stieltjesSamplesC(x,d)
tiny=10*realmin;
huge=.1*realmax;
N = length(x);%Number of samples
xw = [x,ones(N,1)/N];%Weights ( in this case all are 1)

%Compute alpha_0 and beta_0
s0=sum(xw(:,2));
ab(1,1)=xw(:,1)'*xw(:,2)/s0; ab(1,2)=s0; %Compute alpha_0 and beta_0 (alpha_0 = 1/N sum(x) = mean of data, beta_0 = 1)
if d==1, return, end
p1=zeros(N,1); p2=ones(N,1);

for k=1:d-1
  p0=p1; p1=p2; %Assign recurrent polynomials for next iteration
  p2=(xw(:,1)-ab(k,1)).*p1-ab(k,2)*p0;%Three term relation where p2 = pk+1; p1 = pk and p0 = pk-1. xw(:,1) is the independent variable x evaluated at the quadrature points
  s1=xw(:,2)'*(p2.^2);% denominator of alpha_k --> inner product <pk,pk>_dL = int_support(pk(x)^2)dL where dL is the probability measure
  s2=xw(:,1)'*(xw(:,2).*(p2.^2));% numerator of alpha_k --> inner product <x*pk,pk>_dL = int_support(x*pk(x)^2)dL where dL is the probability measure  
  if(max(abs(p2))>huge)|(abs(s2)>huge)
    error(sprintf('impending overflow in stieltjes for k=%3.0f',k))
  end
  if abs(s1)<tiny
    error(sprintf('impending underflow in stieltjes for k=%3.0f',k))
  end
  ab(k+1,1)=s2/s1; ab(k+1,2)=s1/s0;
  s0=s1;% denominator of beta_k for next iteration so pk becomes pk-1 --> inner product <pk,pk>_dL = int_support(pk(x)^2)dL where dL is the probability measure
end