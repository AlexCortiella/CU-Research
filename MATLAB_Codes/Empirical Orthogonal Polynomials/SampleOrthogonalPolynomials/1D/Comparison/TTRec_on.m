function OPx = TTRec_on(x,ab)

N = size(ab,1);
M = length(x);
OPx = zeros(N,M);
OPx(1,:) = ones(1,M);

alpha = ab(:,1);
beta = ab(:,2);

for k = 1:N-1
    
    if k == 1
        OPx(k+1,:) = 1/beta(k)*(x-alpha(k)).*OPx(k,:);
    else
        OPx(k+1,:) = 1/beta(k)*(x-alpha(k)).*OPx(k,:) - beta(k-1)/beta(k).*OPx(k-1,:);
    end
    
end

end