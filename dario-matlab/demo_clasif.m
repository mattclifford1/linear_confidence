% DEMO_CLASIF

% Using simple bounds to optimize the boundary of a linear classifier
% normal to the segment joining the means in feature space.
% TODO: Extend to the non-separable case
%   Idea: Brute force?
% TODO: Show that projecting into the vector joining the empirical means is
% OK. Bound the number of errors w.r.t. projecting in the segment joining
% the real means

% Darío García, May 2010


width=0.3;

% Generate some data
N1=150;
N2=60;

Ntest(1)=1000;
Ntest(2)=1000;

labels_test=ones(sum(Ntest),1);
labels_test(1:Ntest(1))=-1;

labels=-1*ones(N1+N2,1);
labels(1:N1)=1;


idx_1=find(labels==1);
idx_2=find(labels==-1);


mu{1}=[0 0;3 1]';
mu{2}=[5 5;0 5]';
noise_var{1}=0.5*ones(2,1);
noise_var{2}=0.5*ones(2,1);
weights{1}=[0.5 0.5];
weights{2}=[0.5 0.5];

Xtrain=sample_mog((labels+1)/2+1,1,mu,noise_var,weights);
Xtrain=[Xtrain{:}];

Xtest=sample_mog((labels_test+1)/2+1,1,mu,noise_var,weights);
Xtest=[Xtest{:}];


% Large margin classifier 

W=calckernel('rbf',width,Xtrain',Xtrain');        

Wtest=calckernel('rbf',width,Xtrain',Xtest');

% Center the kernel matrix around the mean of the first class
N=size(W,1);
alpha=zeros(N1+N2,1);
alpha(1:N1)=1/N1;
Wc=W-repmat(W*alpha,1,N)-repmat(alpha'*W,N,1)+alpha'*W*alpha*ones(N);
W=Wc;

L=N1+N2;

W1=W(idx_1,idx_1);
Wcross=W(idx_1,idx_2);
W2=W(idx_2,idx_2);

dist=W*labels; % Projection of the data points in the segment joining the two means
D=mean(dist(idx_2))-mean(dist(idx_1));

% Obtain D
%D=sqrt(1/N1^2*sum(W1(:))-2/(N1*N2)*sum(Wcross(:))+1/N2^2*sum(W2(:)));

% Obtain the margin
% TODO: Handle the non-separable case

M=min(dist(idx_1))-max(dist(idx_2))

% Equal margin
b=min(dist(idx_1))-M/2;

% Redistribute the margin
% Solve the optimization problem in terms of delta1
C1=1/2;
C2=1/2;
R=1;
[X,fx,i]=minimize(0.5,@(x) min_func(C1,C2,N1,N2,x,M,R),1e3);
bound=coste(C1,C2,N1,N2,X,M,R);
%keyboard
% Get delta2 from delta1
delta1=X;
delta2=solve_delta(N1,X,N2,M,R);
% Get the margins for each class
R1=R_from_delta(N1,X);
R2=R_from_delta(N2,delta2);
R1+R2-M
% So the boundary is:
b_delta=min(dist(idx_1))-R1;


%%% 
% TEST
%%%
Wtest=calckernel('rbf',width,[Xtrain Xtest]');

% Center the kernel matrix around the mean of the first class
N=size(Wtest,1);
alpha=zeros(N1+N2+sum(Ntest),1);
alpha(1:N1)=1/N1;
Wct=Wtest-repmat(Wtest*alpha,1,N)-repmat(alpha'*Wtest,N,1)+alpha'*Wtest*alpha*ones(N);
Wtest=Wct;  

dist_test=Wtest*[labels;zeros(sum(Ntest),1)];

% Using standard max-margin boundary
labels_test_mm=ones(sum(Ntest),1);
labels_test_mm(dist_test(N1+N2+1:end)<b)=-1;
err_mm=mean(labels_test_mm~=labels_test);
scatter(Xtest(1,:),Xtest(2,:),10,labels_test_mm+2)
title('Using Max-Margin Boundary');

% Using optimized boundary
labels_test_delta=ones(sum(Ntest),1);
labels_test_delta(dist_test(N1+N2+1:end)<b_delta)=-1;
err_delta=mean(labels_test_delta~=labels_test);
figure;
hold on
% Labeled test points
scatter(Xtest(1,:),Xtest(2,:),10,labels_test_delta+2)
% Boundary

title('Using Optimized Boundary');
hold off

% Compare with a SVM
C=10;
Y=[ones(N1,1);-1*ones(N2,1)];
options=sprintf('-s 0 -t 2 -c %i -g %f',C,width);
model=svmtrain(Y,Xtrain',options);
K=calckernel('rbf',width,model.SVs,Xtest');
pred=K*model.sv_coef;
labels_svm=(pred>0)+1;
% Empirical error
err_svm=mean(labels_test.*pred<0);
figure;
scatter(Xtest(1,:),Xtest(2,:),10,labels_svm)
title('Using SVM');

% TODO: Project onto the SVM-chosen direction and use my idea to select
% the bias 


        
fprintf('Error using max-margin: %.3f%%\nError using optimized boundary: %.3f%%\nExpected upper bound: %.3f%%\nError using a SVM: %.3f%%\n',err_mm*100,err_delta*100,bound*100,err_svm*100);
%fprintf('Error using max-margin: %f',err_mm);
