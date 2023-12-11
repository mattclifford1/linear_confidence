function a=deriva_delta(N2,x,N1,M,R)

a=exp(-N1/2*(M/(2*R)-2*(sqrt(1/N1)+sqrt(1/N2))-sqrt(2)*sqrt(log(1/x)/N2))^2)*(-N1*(M/(2*R)-2*(sqrt(1/N1)+sqrt(1/N2))-sqrt(2)*sqrt(log(1/x)/N2))*1/(N2*x)*1/(sqrt(2)*sqrt(log(1/x)/N2)));
