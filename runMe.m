% Clear all if needed
clear all;
close all;
clc;

% Parameters setup

lambda = 1.0; % set lambda of the SVM fittig. Lambda is the regularization parameter of the SVM that multiply the hinge loss function

Kmax = 500; % Maximum number of iteration indicates the acceptable value within which convergence should be achieved to avoid forced stop. Should be more than enough to reach convergence.

rho = 1.0; % Set rho parameter of the augmented Lagrangian. It is part of the regularization term added for obtaining a strictly convex optimization problem

[trainSamples,trainLabels, testSamples, testLabels] = newData("load"); % if argument="load" it loads, else for any other argument e.g. "Random" it generates new data collected in Dataset. p indicates the partitioned sets

m = size(trainSamples,1); % extract training samples number
Nss = floor(m.*0.1); % number of sub sets to create for the split by data approach

% Calculate sub-partitions that will be assigned to each agent
p = zeros(1,m);
p(trainLabels == 1)  = sort(randi([1, floor(Nss/2)], sum(trainLabels==1),1));
p(trainLabels == -1) = sort(randi([floor(Nss/2)+1, Nss], sum(trainLabels==-1),1));

% Let's call the function to solve the SVM problem using ADMM
[results] = svm_admm(trainSamples, trainLabels, lambda, Kmax, p, rho);

% Now let's see the results
K = length(results.objval); % Retrieve all steps' data

% Plot the resulting values for each iteration
figure;
plot(1:K, results.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');

% Plot performance trend as the iterations pass
figure;
subplot(2,1,1);
semilogy(1:K, max(1e-8, results.r_norm), 'k', 1:K, results.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(2,1,2);
semilogy(1:K, max(1e-8, results.s_norm), 'k', 1:K, results.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)');

% Show in figure the data and the decision boundary
figure;
hold on;
gscatter(trainSamples(:,1),trainSamples(:,2), trainLabels);

xavg = mean(results.lastx,2);

xspMax = max(max( max(trainSamples), max(testSamples)));
xspMin = min(min( min(trainSamples), min(testSamples)));
xsp = linspace(xspMin,xspMax);

g = @(xsp) -(xsp*xavg(1) + xavg(3))/xavg(2); % xavg is  [w,b]
yg = g(xsp);
plot(xsp,yg,'b--','LineWidth',2,'DisplayName','Boundary SVM1')

% Now we will use matlab fitcsvm to train an SVM and test its performace
svm2 = fitcsvm(trainSamples,trainLabels);
cvMdl = crossval(svm2); % Performs stratified 10-fold cross-validation
cvtrainError = kfoldLoss(cvMdl);
cvtrainAccuracy = 1-cvtrainError

newError = loss(svm2,testSamples,testLabels);
newAccuracy = 1-newError

% plot svm2 boundary
f = @(xsp) -(xsp*svm2.Beta(1) + svm2.Bias)/svm2.Beta(2);
yp=f(xsp);
plot(xsp,yp,'g--','LineWidth',2,'DisplayName','Boundary SVM2');
hold off;

%THE PART OF THE CLASSIFICATION TEST ON DATA NOT SEEN BEFORE IS ALSO MISSING. (ONE THING AT A TIME)