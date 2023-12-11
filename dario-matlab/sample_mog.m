function X=sample_mog(labels,L,mu,sigma,weights)
N=length(labels);

if (isscalar(L))
    L=L*ones(N,1);
end

for i=1:N
%    ang=2*pi*rand(L(i),1);
%    [a,b]=pol2cart(ang,R(labels(i)));
    X{i}=mixgauss_sample(mu{labels(i)}, sigma{labels(i)}, weights{labels(i)}, L(i));
%    X{i}=repmat(mu(:,labels(i)),1,L(i))+randn(2,L(i))*sqrt(sigma(labels(i)));
end
