function c=coste(C1,C2,N1,N2,x,M,R)
N=length(x);
c=-1*ones(N,1);

for i=1:N
    x2=solve_delta(N1,x(i),N2,M,R);
    if (~isreal(x2))
        error('Complex delta2');
    end
    c(i)=C1*( (1-x(i))*1/(N1+1)+x(i))+C2*( (1-x2)*1/(N1+1)+x2);
end