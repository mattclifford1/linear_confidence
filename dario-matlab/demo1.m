% DEMO LEARN DELTA
N1=100;
N2=50;
C1=1;
C2=1;
M=2.4;
R=1;


% Solve the optimization problem in terms of delta1
[X,fx,i]=minimize(0.5,@(x) min_func(C1,C2,N1,N2,x,M,R),1e3);
% Get delta2 from delta1
delta2=solve_delta(N1,X,N2,M,R);
% Get the margins for each class
R1=R_from_delta(N1,X)
R2=R_from_delta(N2,delta2)
% Check that they equal the total margin
R1+R2
