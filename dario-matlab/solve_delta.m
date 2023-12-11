function d=solve_delta(N2,delta2,N1,M,R)

N_DELTA=length(delta2);
N_M=length(M);

d=-1*ones(N_DELTA,N_M);

for i=1:N_DELTA
    for j=1:N_M
        d(i,j)=exp(-(N1/2)*(M(j)/2-2*(sqrt(1/N1)+sqrt(1/N2))-sqrt(2*log(1/delta2(i))/N2))^2);
    end
end

d=squeeze(d);