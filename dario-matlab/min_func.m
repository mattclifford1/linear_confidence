function [v d]=min_func(C1,C2,N1,N2,x,M,R);
v=coste(C1,C2,N1,N2,x,M,R);
d=deriva_coste(C1,C2,N1,N2,x,M,R);
%dx=1e-3;
%(coste(C1,C2,N1,N2,x+dx,M,R)-(coste(C1,C2,N1,N2,x,M,R)+d*dx))/v