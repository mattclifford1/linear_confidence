function a=deriva_coste(C1,C2,N1,N2,x,M,R)

a=C1*(N1/(N1+1))+deriva_delta(N1,x,N2,M,R)*C2*(N2/(N2+1));
