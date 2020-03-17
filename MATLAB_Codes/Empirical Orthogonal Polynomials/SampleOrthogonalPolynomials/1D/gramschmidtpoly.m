function z = gramschmidtpoly(x,N,M)

for i=1:N
s(i,:)=x(i,:);
end

e(1)=s(1,:)*conj(s(1,:).');
phi(1,:)=s(1,:)/sqrt(e(1));

for i=2:N
    th(i,:)=zeros(1,M);
    for r=i-1:-1:1
        th(i,:)=th(i,:)+(s(i,:)*conj(phi(r,:).'))*phi(r,:);
    end
    th(i,:)=s(i,:)-th(i,:);
    e(i)=th(i,:)*conj(th(i,:).');
    phi(i,:)=th(i,:)/sqrt(e(i));
end

z=phi(1:N,:);

end

