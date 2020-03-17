% This script evaluated a 1-d Legendre polynomial on [-1,1]

% p is the maximum order of the PC (total order)

function Monomial = my_monomial_1d(p,x)

Monomial = zeros(p+1,size(x,2));

if p==0 
    Monomial(p+1,:) = ones(1,size(x,2));
elseif p==1
    Monomial(p  ,:) = ones(1,size(x,2));
    Monomial(p+1,:) = x;
else 
    Monomial(1,:) = ones(1,size(x,2));
    Monomial(2,:) = x;
    for ord = 2:p
        Monomial(ord+1,:)= x.* Monomial(ord+1-1,:);
    end
end


        

