% Clear all if needed
clear all;
close all;
clc;

% Parameters setup
worstAssign = false; % False: assign samples randomically to each agent. True: Assign them mixed samples' blocks with the same label to make convergence harder
lambda = 6e-1; % Set lambda of the SVM fittig. 
Kmax = 500; % Maximum number of iteration indicates the acceptable value within which convergence should be achieved to avoid forced stop.
rho = 1e0; % Set rho parameter, part of the regularization term added for obtaining a strictly convex optimization problem

% Load dataset from a file or generate one
[trainSamples,trainLabels, trainA, testSamples, testLabels] = newData("load"); % load: use a dataset file, else for any other argument e.g. "Random" it generates a new random dataset.

m = size(trainA,1); % Extract training samples number
Nss = floor(m.*0.1); % Number of sub sets to create for the split by data approach, each subset is assigned to an agent

% Calculate sub-partitions that will be assigned to each agent
p = zeros(1,m);
if(worstAssign)
    p(trainLabels == 1)  = sort(randi([1, floor(Nss/2)], sum(trainLabels==1),1));
    p(trainLabels == -1) = sort(randi([floor(Nss/2)+1, Nss], sum(trainLabels==-1),1));
else
    p = randi([1, Nss],m,1);

end
% Let's call the function to solve the SVM problem using ADMM
[results] = svm_admm(trainA, lambda, Kmax, p, rho);

% Now let's see the results
K = length(results.objval); % Retrieve all steps' data

% Plot the resulting values for each iteration
figure("Name"," Objective function vs iterations");;
plot(1:K, results.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');

% Plot performance trend as the iterations pass
figure("Name","Performance and stop conditions vs iterations");
subplot(2,1,1);
semilogy(1:K, max(1e-8, results.r_norm), 'k', 1:K, results.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(2,1,2);
semilogy(1:K, max(1e-8, results.s_norm), 'k', 1:K, results.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)');

% Show in figure the data and the decision boundary of svm_admm (svm1)
figure("Name","Training Samples vs Boundaries");
hold on;
gscatter(trainSamples(1,:),trainSamples(2,:), trainLabels);

xavg = mean(results.lastx,2);

xspMax = max(max( max(trainSamples,[],2), max(testSamples,[],2)));
xspMin = min(min( min(trainSamples,[],2), min(testSamples,[],2)));
xsp = linspace(xspMin,xspMax);

g = @(xsp) -(xsp*xavg(1) +xavg(3))/xavg(2); % xavg is  [w,b]
yg = g(xsp);
plot(xsp,yg,'b--','LineWidth',2,'DisplayName','Boundary SVM1')

% SVM_ADMM accuracy
svm1TestAccuracy = length(find(testLabels==sign(xavg(1:2,:)'*testSamples+xavg(3))))/size(testSamples,2)

% Now we will use matlab fitcsvm to train an SVM and test its performace (svm2)
svm2 = fitcsvm(trainSamples',trainLabels');

svm2TestError = loss(svm2,testSamples',testLabels');
svm2TestAccuracy = 1-svm2TestError

% plot svm2 boundary
f = @(xsp) -(xsp*svm2.Beta(1) + svm2.Bias)/svm2.Beta(2);
yp=f(xsp);
plot(xsp,yp,'g--','LineWidth',2,'DisplayName','Boundary SVM2');
hold off;

figure("Name","Test Samples vs Boundaries");
hold on;
% Plot also test samples
gscatter(testSamples(1,:),testSamples(2,:), testLabels);
plot(xsp,yg,'b--','LineWidth',2,'DisplayName','Boundary SVM1')
plot(xsp,yp,'g--','LineWidth',2,'DisplayName','Boundary SVM2');
hold off;
   