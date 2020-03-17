function Ck = menger2(P1,P2,P3)

%Compute Ecuclidean distances j < k < l (j = k -1; l = k + 1)
u = P1 - P2;
v = P3 - P2;

Ck = 2*(v(1)*u(2) - u(1)*v(2))/(norm(u)*norm(v)*norm(u-v));% Ck = 2*sin(xyz)/(|x - z|)

end

