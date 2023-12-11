function f=evaluate_delta(delta1,N1,delta2,N2,M,R)

f=sqrt(1/N1)*(2+sqrt(2*log(1/delta1)))+sqrt(1/N2)*(2+sqrt(2*log(1/delta2)))-M/(2*R);
