%% CSAPI TEST

xx = linspace(0,6,121);
x = [2 3 5];
y = [1 0 4];
spl = csapi(x,y,xx);
plot(xx,spl,'k-',x,y,'ro')
title('Interpolant to Three Points')