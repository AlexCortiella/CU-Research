function Ck = menger(P1,P2,P3)

%Compute Ecuclidean distances j < k < l (j = k -1; l = k + 1)
xi = [P1(1),P2(1),P3(1)];
eta = [P1(2),P2(2),P3(2)];

Pjk = (xi(2) - xi(1))^2 + (eta(2) - eta(1))^2;% P12
Pkl = (xi(3) - xi(2))^2 + (eta(3) - eta(2))^2;% P23
Plj = (xi(2) - xi(1))^2 + (eta(2) - eta(1))^2;% P31

%Compute curvature at k given 3 points
num = 2*(xi(1)*eta(2) + xi(2)*eta(3) + xi(3)*eta(1) - xi(1)*eta(3) - xi(2)*eta(1) - xi(3)*eta(2))
den = sqrt(Pjk*Pkl*Plj)

Ck =  num/den;

end

